// Hair Game — webcam face tracking (Apple Vision) + 3D physics hair you can shave, grow, brush, curl and dye.
//
// • Bald effect: Vision person segmentation + face landmarks + a per-pixel skin/hair classifier find the
//   real hair. Hair over the skull becomes a shaded bald scalp tinted from your forehead; hair outside the
//   skull is replaced by a background plate learned while you move. Face and brows are protected.
// • Hair: guide strands are simulated in 3D (Verlet + follow-the-leader) around an ellipsoid skull and
//   face, then rendered as clumps of child strands with curls, shadow and sheen.
// • Rendering: camera, overlay and hair are Core Animation layers (GPU composited), so the main thread
//   only builds paths.
import Cocoa
import AVFoundation
import Vision
import CoreImage
import Metal
import QuartzCore

// MARK: - Math

extension CGPoint {
    @inline(__always) func z(_ p: V3, _ S: CGFloat) -> CGFloat { p.z / S }
    static func + (a: CGPoint, b: CGPoint) -> CGPoint { CGPoint(x: a.x + b.x, y: a.y + b.y) }
    static func - (a: CGPoint, b: CGPoint) -> CGPoint { CGPoint(x: a.x - b.x, y: a.y - b.y) }
    static func * (a: CGPoint, s: CGFloat) -> CGPoint { CGPoint(x: a.x * s, y: a.y * s) }
    var len: CGFloat { hypot(x, y) }
    var norm: CGPoint { let l = len; return l > 1e-6 ? self * (1 / l) : CGPoint(x: 0, y: -1) }
}

struct V3 {
    var x: CGFloat, y: CGFloat, z: CGFloat
    init(_ x: CGFloat, _ y: CGFloat, _ z: CGFloat) { self.x = x; self.y = y; self.z = z }
    static let zero = V3(0, 0, 0)
    static func + (a: V3, b: V3) -> V3 { V3(a.x + b.x, a.y + b.y, a.z + b.z) }
    static func - (a: V3, b: V3) -> V3 { V3(a.x - b.x, a.y - b.y, a.z - b.z) }
    static func * (a: V3, s: CGFloat) -> V3 { V3(a.x * s, a.y * s, a.z * s) }
    var len: CGFloat { (x * x + y * y + z * z).squareRoot() }
    var norm: V3 { let l = len; return l > 1e-6 ? self * (1 / l) : V3(0, -1, 0) }
    var xy: CGPoint { CGPoint(x: x, y: y) }
}

func lerpAngle(_ a: CGFloat, _ b: CGFloat, _ t: CGFloat) -> CGFloat {
    var d = b - a
    while d > .pi { d -= 2 * .pi }
    while d < -.pi { d += 2 * .pi }
    return a + d * t
}
@inline(__always) func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t = min(1, max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t)
}

struct RGB: Hashable {
    var r: CGFloat, g: CGFloat, b: CGFloat
    init(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) { self.r = r; self.g = g; self.b = b }
    init(_ c: NSColor) {
        let s = c.usingColorSpace(.sRGB) ?? c
        r = s.redComponent; g = s.greenComponent; b = s.blueComponent
    }
    func mix(_ o: RGB, _ t: CGFloat) -> RGB { RGB(r + (o.r - r) * t, g + (o.g - g) * t, b + (o.b - b) * t) }
    func scaled(_ s: CGFloat) -> RGB { RGB(min(1, r * s), min(1, g * s), min(1, b * s)) }
    func cg(_ a: CGFloat = 1) -> CGColor { CGColor(srgbRed: r, green: g, blue: b, alpha: a) }
    var quantized: RGB { RGB((r * 24).rounded() / 24, (g * 24).rounded() / 24, (b * 24).rounded() / 24) }
}

/// One Euro filter: heavy smoothing when still, low lag when moving (Casiez et al. 2012).
struct OneEuro {
    var minCutoff: Double, beta: Double, dCutoff = 1.0
    private var x: Double?, dx = 0.0
    init(_ minCutoff: Double, _ beta: Double) { self.minCutoff = minCutoff; self.beta = beta }
    private func alpha(_ cutoff: Double, _ dt: Double) -> Double { let tau = 1 / (2 * .pi * cutoff); return 1 / (1 + tau / dt) }
    mutating func reset() { x = nil; dx = 0 }
    mutating func callAsFunction(_ v: Double, dt: Double) -> Double {
        guard let px = x, dt > 0 else { x = v; return v }
        dx += alpha(dCutoff, dt) * ((v - px) / dt - dx)
        let nx = px + alpha(minCutoff + beta * abs(dx), dt) * (v - px)
        x = nx
        return nx
    }
}

/// Launch environment, read once (`ProcessInfo.environment` builds a new dictionary on every access).
enum Env {
    static let vars = ProcessInfo.processInfo.environment
    static let debug = vars["HAIRGAME_DEBUG"] != nil
    /// Replay, synthetic and self-test runs: never pause when the window is hidden (they run unattended).
    static let harness = vars["HAIRGAME_REPLAY"] != nil || vars["HAIRGAME_SYNTH"] != nil || vars["HAIRGAME_SELFTEST"] != nil
}

/// Per-stage timing (enabled with HAIRGAME_PERF). Thread-safe; reports avg / max ms per stage.
enum Perf {
    static let on = Env.vars["HAIRGAME_PERF"] != nil
    private static let lock = NSLock()
    private static var acc: [String: (sum: Double, max: Double, n: Int)] = [:]
    static func add(_ k: String, _ ms: Double) {
        guard on else { return }
        lock.lock(); var e = acc[k] ?? (0, 0, 0); e.sum += ms; e.max = max(e.max, ms); e.n += 1; acc[k] = e; lock.unlock()
    }
    @inline(__always) static func time<T>(_ k: String, _ f: () -> T) -> T {
        guard on else { return f() }
        let t = CACurrentMediaTime(); let r = f(); add(k, (CACurrentMediaTime() - t) * 1000); return r
    }
    static func report() {
        guard on else { return }
        lock.lock(); let snap = acc; acc = [:]; lock.unlock()
        let line = snap.keys.sorted().map { k in let e = snap[k]!; return String(format: "%@ %.1f/%.1f(n%d)", k, e.sum / Double(max(1, e.n)), e.max, e.n) }.joined(separator: "  ")
        FileHandle.standardError.write("PERF \(line)\n".data(using: .utf8)!)
    }
}

// MARK: - Face geometry

/// Landmarks in normalized image coordinates (origin bottom-left, unmirrored).
struct Landmarks { var eyeA, eyeB: CGPoint; var brows, contour, nose, lips: [CGPoint]; var noseTip: CGPoint? = nil; var vYaw: CGFloat? = nil; var vPitch: CGFloat? = nil }
extension Landmarks {
    /// Distance from the eye midpoint to the mouth corners' midpoint. Unlike eye spacing it barely changes when the
    /// head turns, so it gives the face size on strong turns (where Vision squeezes the eyes together).
    func mouthDrop(_ toPx: (CGPoint) -> CGPoint) -> CGFloat? {
        guard lips.count >= 4 else { return nil }
        let l = lips.map(toPx), a = toPx(eyeA), b = toPx(eyeB)
        let dir = CGPoint(x: b.x - a.x, y: b.y - a.y), n = max(1e-6, hypot(dir.x, dir.y))
        let proj = l.map { ($0.x - a.x) * dir.x / n + ($0.y - a.y) * dir.y / n }
        guard let lo = proj.indices.min(by: { proj[$0] < proj[$1] }), let hi = proj.indices.max(by: { proj[$0] < proj[$1] }) else { return nil }
        let m = CGPoint(x: (l[lo].x + l[hi].x) / 2, y: (l[lo].y + l[hi].y) / 2)
        return hypot(m.x - (a.x + b.x) / 2, m.y - (a.y + b.y) / 2)
    }
}
/// How much the turn-robust size cue (mouth drop) takes over from eye spacing, by turn angle.
@inline(__always) func dropWeight(_ yaw: CGFloat) -> CGFloat { let t = max(0, min(1, (abs(yaw) - 0.45) / 0.35)); return t * t * (3 - 2 * t) }

/// Face-local frame: origin between the eyes, unit = inter-eye distance, x along the eyes, z toward camera.
/// The head model is rotated by yaw (turn) and pitch (nod) about the eyes, then rolled and scaled into view.
struct LocalFrame {
    var origin: CGPoint; var scale: CGFloat; var roll: CGFloat
    var camDist: CGFloat = 1e6                               // camera distance from the eyes, in eye-widths (perspective)
    private(set) var yaw: CGFloat = 0, pitch: CGFloat = 0
    static let pivotZ: CGFloat = 0.6                         // depth of the eyes in the head model
    private var cr: CGFloat = 1, sr: CGFloat = 0, cy: CGFloat = 1, sy: CGFloat = 0, cp: CGFloat = 1, sp: CGFloat = 0

    init(origin: CGPoint, scale: CGFloat, roll: CGFloat, yaw: CGFloat = 0, pitch: CGFloat = 0) {
        self.origin = origin; self.scale = scale; self.roll = roll
        setRotation(roll: roll, yaw: yaw, pitch: pitch)
    }
    init(_ a: CGPoint, _ b: CGPoint) {
        let (l, r) = a.x < b.x ? (a, b) : (b, a)
        self.init(origin: (a + b) * 0.5, scale: max(1, (r - l).len), roll: atan2(r.y - l.y, r.x - l.x))
    }
    mutating func setRotation(roll: CGFloat, yaw: CGFloat, pitch: CGFloat) {
        self.roll = roll; self.yaw = yaw; self.pitch = pitch
        cr = cos(roll); sr = sin(roll); cy = cos(yaw); sy = sin(yaw); cp = cos(pitch); sp = sin(pitch)
    }
    func world(_ p: CGPoint) -> CGPoint {
        CGPoint(x: origin.x + (p.x * cr - p.y * sr) * scale, y: origin.y + (p.x * sr + p.y * cr) * scale)
    }
    func local(_ p: CGPoint) -> CGPoint {
        let d = p - origin
        return CGPoint(x: (d.x * cr + d.y * sr) / scale, y: (-d.x * sr + d.y * cr) / scale)
    }
    @inline(__always) func rotate(_ p: V3) -> V3 {           // pitch about x, then yaw about y
        let y1 = p.y * cp - p.z * sp, z1 = p.y * sp + p.z * cp
        return V3(p.x * cy + z1 * sy, y1, -p.x * sy + z1 * cy)
    }
    @inline(__always) func unrotate(_ p: V3) -> V3 {
        let x = p.x * cy - p.z * sy, z1 = p.x * sy + p.z * cy
        return V3(x, p.y * cp + z1 * sp, -p.y * sp + z1 * cp)
    }
    /// Perspective magnification for a point `z` eye-widths in front of the eyes.
    @inline(__always) func persp(_ z: CGFloat) -> CGFloat { camDist / max(camDist * 0.3, camDist - z) }
    func world3(_ p: V3) -> V3 {
        let q = rotate(V3(p.x, p.y, p.z - LocalFrame.pivotZ))
        let k = persp(q.z)
        let w = world(CGPoint(x: q.x * k, y: q.y * k))
        return V3(w.x, w.y, (q.z + LocalFrame.pivotZ) * scale)
    }
    func local3(_ p: V3) -> V3 {
        let l = local(p.xy), zr = p.z / scale - LocalFrame.pivotZ, k = persp(zr)
        let q = unrotate(V3(l.x / k, l.y / k, zr))
        return V3(q.x, q.y, q.z + LocalFrame.pivotZ)
    }
    func worldDir(_ d: V3) -> V3 {
        let q = rotate(d)
        return V3(q.x * cr - q.y * sr, q.x * sr + q.y * cr, q.z)
    }
    /// Inverse of world3 for a point lying on `surface`'s front: finds head-model (x, y) that projects to view point v.
    func unproject(_ v: CGPoint, onto surface: Ellipsoid) -> CGPoint {
        let target = local(v)
        // projection of a surface point (x, y) into flat local coordinates; off the surface, clamp to its rim
        func proj(_ m: CGPoint) -> CGPoint {
            let qx = (m.x - surface.c.x) / surface.r.x, qy = (m.y - surface.c.y) / surface.r.y
            let d = max(0, 1 - qx * qx - qy * qy)
            return local(world3(V3(m.x, m.y, surface.c.z + surface.r.z * d.squareRoot())).xy)
        }
        // damped Newton with a numeric Jacobian (the surface depth makes the mapping non-linear)
        // the mapping can fold near the rim (visible jaw vs hidden underside): coarse search over visible
        // surface points first, then Newton refinement
        func visible(_ m: CGPoint) -> Bool {
            guard let q = surface.front(m.x, m.y) else { return false }
            return rotate(surface.normal(q)).z > 0.05
        }
        let g0 = CGPoint(x: (target.x - 0.2 * sy) / max(0.3, cy), y: target.y)
        var m = g0, best = CGFloat.greatestFiniteMagnitude
        for i in -8...8 { for j in -8...8 {
            let c = CGPoint(x: g0.x + CGFloat(i) * 0.08, y: g0.y + CGFloat(j) * 0.08)
            guard visible(c) else { continue }
            let err = (proj(c) - target).len
            if err < best { best = err; m = c }
        } }
        let e: CGFloat = 1e-3
        for _ in 0..<12 {
            let f0 = proj(m) - target
            if f0.len < 1e-5 { break }
            let fx = (proj(CGPoint(x: m.x + e, y: m.y)) - proj(m)) * (1 / e)
            let fy = (proj(CGPoint(x: m.x, y: m.y + e)) - proj(m)) * (1 / e)
            let det = fx.x * fy.y - fy.x * fx.y
            guard abs(det) > 1e-6 else { break }
            var step = CGPoint(x: (fy.y * f0.x - fy.x * f0.y) / det, y: (-fx.y * f0.x + fx.x * f0.y) / det)
            if step.len > 0.3 { step = step * (0.3 / step.len) }
            let next = m - step
            if !visible(next) { break }
            m = next
        }
        return m
    }
    /// Direction toward the camera, expressed in head-model space.
    var viewDirLocal: V3 { unrotate(V3(0, 0, 1)) }
}

/// Head pose cues: the nose tip's position relative to the midpoint between the eyes (flat local units).
/// The tip sits ~0.45 eye-widths in front of the eyes, so turning moves it sideways (x ≈ 0.45·tan yaw)
/// and nodding moves it up/down. Faces are symmetric, so x needs no calibration.
func poseCues(_ lm: Landmarks, _ f: LocalFrame, _ map: (CGPoint) -> CGPoint) -> (x: CGFloat, y: CGFloat)? {
    guard let tip = lm.noseTip else { return nil }
    let t = f.local(map(tip))
    guard abs(t.x) < 1.5, t.y < -0.2, t.y > -1.5 else { return nil }
    return (t.x, t.y)
}

struct Ellipsoid {
    var c: V3, r: V3
    func q(_ p: V3) -> V3 { V3((p.x - c.x) / r.x, (p.y - c.y) / r.y, (p.z - c.z) / r.z) }
    /// Point on the front surface above (x, y), or nil if outside the silhouette.
    func front(_ x: CGFloat, _ y: CGFloat) -> V3? {
        let qx = (x - c.x) / r.x, qy = (y - c.y) / r.y
        let d = 1 - qx * qx - qy * qy
        return d < 0 ? nil : V3(x, y, c.z + r.z * d.squareRoot())
    }
    func normal(_ p: V3) -> V3 { let q = q(p); return V3(q.x / r.x, q.y / r.y, q.z / r.z).norm }
    /// True if the ray from p along d (toward the camera) passes through this ellipsoid.
    func occludes(_ p: V3, _ d: V3) -> Bool {
        let o = V3((p.x - c.x) / r.x, (p.y - c.y) / r.y, (p.z - c.z) / r.z), v = V3(d.x / r.x, d.y / r.y, d.z / r.z)
        let cc = o.x * o.x + o.y * o.y + o.z * o.z - 1
        if cc < 0 { return true }
        let b = o.x * v.x + o.y * v.y + o.z * v.z
        if b >= 0 { return false }
        return b * b - (v.x * v.x + v.y * v.y + v.z * v.z) * cc > 0
    }
    func inSilhouette(_ x: CGFloat, _ y: CGFloat) -> Bool { let qx = (x - c.x) / r.x, qy = (y - c.y) / r.y; return qx * qx + qy * qy < 1 }
    /// Push a point out to the surface (inflated by `pad`).
    func pushOut(_ p: V3, pad: CGFloat) -> V3? {
        let rr = V3(r.x * pad, r.y * pad, r.z * pad)
        let q = V3((p.x - c.x) / rr.x, (p.y - c.y) / rr.y, (p.z - c.z) / rr.z)
        let d2 = q.x * q.x + q.y * q.y + q.z * q.z
        guard d2 < 1, d2 > 1e-8 else { return nil }
        let k = 1 / d2.squareRoot()
        return V3(c.x + q.x * k * rr.x, c.y + q.y * k * rr.y, c.z + q.z * k * rr.z)
    }
}

/// Measurements of a face in local units.
struct FaceShape {
    var browY: CGFloat = 0.45, browBottom: CGFloat = 0.3, browHalf: CGFloat = 0.95
    var chinY: CGFloat = -2.0, noseY: CGFloat = -0.85
    var lipTop: CGFloat = -1.15, lipBottom: CGFloat = -1.5, mouthHalf: CGFloat = 0.45, halfWidth: CGFloat = 1.0
    var upperLip: [CGPoint] = []      // top edge of the upper lip, sorted by x (local units)
    var contour: [CGPoint] = []       // jawline from temple to temple (local units)
    var beardCenter: CGPoint?         // middle of the lower face, in the same mapping as `contour`
    var edgeBrowY: CGFloat?           // brow line in the same mapping as `contour`

    init() {}
    init(_ lm: Landmarks, _ f: LocalFrame, _ map: (CGPoint) -> CGPoint) { self.init(lm) { f.local(map($0)) } }
    /// `L` maps a normalized landmark to head-model (x, y). `edge` maps outline points (the jawline lies on the
    /// face's silhouette, not its front surface, so it must not be projected onto the front).
    init(_ lm: Landmarks, edge: ((CGPoint) -> CGPoint)? = nil, _ L: (CGPoint) -> CGPoint) {
        let brows = lm.brows.map(L), nose = lm.nose.map(L), lips = lm.lips.map(L), contour: [CGPoint] = edge.map { e in lm.contour.map(e) } ?? lm.contour.map(L)
        self.contour = contour
        if let e = edge, let nb = lm.nose.min(by: { $0.y < $1.y }), let lt = lm.lips.max(by: { $0.y < $1.y }) {
            beardCenter = (e(nb) + e(lt)) * 0.5
        }
        if let e = edge, !lm.brows.isEmpty { edgeBrowY = lm.brows.map { e($0).y }.max() }
        if let v = brows.map(\.y).max() { browY = v }
        if let v = brows.map(\.y).min() { browBottom = v }
        if let v = brows.map({ abs($0.x) }).max() { browHalf = v }
        if let v = contour.map(\.y).min() { chinY = v }
        if let v = nose.map(\.y).min() { noseY = v }
        if let v = lips.map(\.y).max() { lipTop = v }
        if lips.count >= 6 {   // outer-lip points above the mouth's middle line form the upper lip edge
            let mid = (lips.map(\.y).max()! + lips.map(\.y).min()!) / 2
            upperLip = lips.filter { $0.y >= mid }.sorted { $0.x < $1.x }
        }
        if let v = lips.map(\.y).min() { lipBottom = v }
        if let v = lips.map({ abs($0.x) }).max() { mouthHalf = v }
        if let lo = contour.map(\.x).min(), let hi = contour.map(\.x).max() { halfWidth = (hi - lo) / 2 }   // span, robust to off-centre contours
    }
    mutating func blend(_ o: FaceShape, _ t: CGFloat) {
        func m(_ a: inout CGFloat, _ b: CGFloat) { a += (b - a) * t }
        m(&browY, o.browY); m(&browBottom, o.browBottom); m(&browHalf, o.browHalf); m(&chinY, o.chinY); m(&noseY, o.noseY)
        if let a = edgeBrowY, let b = o.edgeBrowY { edgeBrowY = a + (b - a) * t } else { edgeBrowY = o.edgeBrowY ?? edgeBrowY }
        if let a = beardCenter, let b = o.beardCenter { beardCenter = a + (b - a) * t } else { beardCenter = o.beardCenter ?? beardCenter }
        if contour.count == o.contour.count { contour = zip(contour, o.contour).map { $0 + ($1 - $0) * t } } else { contour = o.contour }
        if upperLip.count == o.upperLip.count { upperLip = zip(upperLip, o.upperLip).map { $0 + ($1 - $0) * t } } else { upperLip = o.upperLip }
        m(&lipTop, o.lipTop); m(&lipBottom, o.lipBottom); m(&mouthHalf, o.mouthHalf); m(&halfWidth, o.halfWidth)
    }
    // 3D head model shared by the bald effect, the hair roots and collisions.
    /// Measured head outline (top, half width; eye-widths) minus hair thickness, when known.
    var skullTop: CGFloat = 0, skullHalfW: CGFloat = 0
    var skull: Ellipsoid {
        let cy = browY + 0.1
        var rx = halfWidth * 1.1, ry: CGFloat = 1.62
        if skullTop > 0 { ry = max(1.1, min(2.3, skullTop - cy)) }
        if skullHalfW > 0 { rx = max(halfWidth * 0.95, min(halfWidth * 1.6, skullHalfW / 0.97)) }
        // a real skull is deeper front-to-back than it is wide: centre well behind the eyes
        return Ellipsoid(c: V3(0, cy, -0.7), r: V3(rx, ry, max(1.55, rx * 1.3)))
    }
    var face: Ellipsoid {
        Ellipsoid(c: V3(0, (browY + 0.1 + chinY) / 2, -0.1), r: V3(halfWidth * 0.97, (browY + 0.1 - chinY) / 2, 1.05))
    }
    /// Left/right edges of the visible face outline at height y (contour space). Above the temples, the outline's ends.
    func faceSpan(atY y: CGFloat) -> (CGFloat, CGFloat)? {
        guard contour.count > 2 else { return nil }
        var xs: [CGFloat] = []
        for i in 1..<contour.count {
            let a = contour[i - 1], b = contour[i]
            if (a.y - y) * (b.y - y) <= 0, abs(b.y - a.y) > 1e-6 { xs.append(a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y)) }
        }
        if xs.count >= 2 { return (xs.min()!, xs.max()!) }
        if y > max(contour.first!.y, contour.last!.y) { return (min(contour.first!.x, contour.last!.x), max(contour.first!.x, contour.last!.x)) }
        return nil
    }
    /// Point on the jawline at arc-length fraction t (0 = one temple, 1 = the other).
    func jawPoint(_ t: CGFloat) -> CGPoint? {
        guard contour.count > 1 else { return nil }
        var lens: [CGFloat] = [0]
        for i in 1..<contour.count { lens.append(lens[i - 1] + (contour[i] - contour[i - 1]).len) }
        let target = max(0, min(1, t)) * lens.last!
        for i in 1..<contour.count where lens[i] >= target {
            let k = (target - lens[i - 1]) / max(1e-4, lens[i] - lens[i - 1])
            return contour[i - 1] + (contour[i] - contour[i - 1]) * k
        }
        return contour.last
    }
    /// Beard anchor: from the jawline at t, `inset` of the way toward the middle of the lower face.
    func beardPoint(_ t: CGFloat, _ inset: CGFloat) -> CGPoint? {
        guard let e = jawPoint(t) else { return nil }
        return e + ((beardCenter ?? CGPoint(x: 0, y: (noseY + lipTop) / 2)) - e) * inset
    }
    /// Height of the upper lip's top edge at x (flat extrapolation past the corners).
    func upperLipY(_ x: CGFloat) -> CGFloat {
        guard let first = upperLip.first, let last = upperLip.last else { return lipTop }
        if x <= first.x { return first.y }
        if x >= last.x { return last.y }
        for i in 1..<upperLip.count where upperLip[i].x >= x {
            let a = upperLip[i - 1], b = upperLip[i]
            return a.y + (b.y - a.y) * (x - a.x) / max(1e-4, b.x - a.x)
        }
        return lipTop
    }
    func hairline(_ x: CGFloat) -> CGFloat { browY + 0.85 - 0.5 * min(1, pow(x / skull.r.x, 2)) }
}

// MARK: - Camera, Vision and the bald effect

let PW = 480, PH = 270   // processing resolution for the bald overlay

struct Frame { var sample: CMSampleBuffer; var pixels: CVPixelBuffer; var overlay: CGImage?; var landmarks: Landmarks? }

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "hairgame.camera", qos: .userInteractive)
    let ci = CIContext(options: [.cacheIntermediates: false])
    let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    var onFrame: ((Frame) -> Void)?
    var onError: ((String) -> Void)?
    var headScale: Float = 1.0
    var headYaw: Float = 0, headPitch: Float = 0   // set by the view each frame (camera space)
    private var camYaw: Float = 0                   // this frame's yaw as used by the bald cap
    private var lastGoodLm: Landmarks?              // previous frame's landmarks (outlier rejection)
    private var lmRejects = 0, lmMisses = 0
    var headDist: Float = 1e6                        // camera distance in eye-widths (perspective)
    var strength: Float = 1.0

    private let segReq: VNGeneratePersonSegmentationRequest = {
        let r = VNGeneratePersonSegmentationRequest()
        r.qualityLevel = .balanced
        r.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return r
    }()
    private var frameBuf = [UInt8](repeating: 0, count: PW * PH * 4)
    private var maskBuf = [UInt8](repeating: 0, count: PW * PH)
    private var plate = [Float](repeating: 0, count: PW * PH * 3)
    private var plateW = [Float](repeating: 0, count: PW * PH)
    /// The plate is stored exposure-normalised: the webcam's auto exposure / white balance drift as the head moves,
    /// and background hidden behind the head for a while would otherwise come back too bright or too dark.
    private var plateGain: (Float, Float, Float) = (1, 1, 1)
    private var alpha = [Float](repeating: 0, count: PW * PH)
    private var tmp = [Float](repeating: 0, count: PW * PH)
    private var protMask = [Float](repeating: 0, count: PW * PH)
    private var prevAlpha: [Float] = []
    private var protBytes = [UInt8](repeating: 0, count: PW * PH)
    private var out = [UInt8](repeating: 0, count: PW * PH * 4)
    private var skin: (Float, Float, Float) = (0.85, 0.66, 0.56)
    private var shape: FaceShape?
    private var hasMask = false
    private let rectReq: VNDetectFaceRectanglesRequest = { let r = VNDetectFaceRectanglesRequest(); r.revision = VNDetectFaceRectanglesRequestRevision3; return r }()
    private let faceReq = VNDetectFaceLandmarksRequest()
    private let segQueue = DispatchQueue(label: "hairgame.seg", qos: .userInteractive)
    private let overlayQueue = DispatchQueue(label: "hairgame.overlay", qos: .userInteractive)
    private let overlayGate = DispatchSemaphore(value: 1)
    private var segFrame = 0
    private var bgCache: [Float] = []
    private var overlayFrame = 0

    func dlog(_ m: String) { if Env.debug { FileHandle.standardError.write("[cam] \(m)\n".data(using: .utf8)!) } }
    func start() {
        dlog("auth status \(AVCaptureDevice.authorizationStatus(for: .video).rawValue)")
        AVCaptureDevice.requestAccess(for: .video) { ok in
            self.dlog("access \(ok)")
            guard ok else {
                DispatchQueue.main.async { self.onError?("Camera access denied — enable it in System Settings › Privacy & Security › Camera") }
                return
            }
            self.queue.async { self.configure() }
        }
    }

    private var observing = false
    private var deviceID: String?
    private func configure() {
        if !observing { observing = true; observeSession() }
        session.beginConfiguration()
        for i in session.inputs { session.removeInput(i) }       // reconfiguring (e.g. camera re-plugged)
        for o in session.outputs { session.removeOutput(o) }
        guard let dev = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            session.commitConfiguration()                        // never leave the session mid-configuration
            dlog("no camera / input failed")
            DispatchQueue.main.async { self.onError?("No camera found — connect a camera, or close apps that are using it") }
            return
        }
        session.addInput(input)
        deviceID = dev.uniqueID
        if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }   // must follow addInput
        let o = AVCaptureVideoDataOutput()
        o.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,   // 720p is plenty and much cheaper
                           kCVPixelBufferWidthKey as String: 1280, kCVPixelBufferHeightKey as String: 720]
        o.alwaysDiscardsLateVideoFrames = true
        o.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(o) { session.addOutput(o) }
        session.commitConfiguration()
        dlog("device \(AVCaptureDevice.default(for: .video)?.localizedName ?? "none") inputs=\(session.inputs.count) outputs=\(session.outputs.count)")
        session.startRunning()
        dlog("running \(session.isRunning)")
    }

    /// Recover from the things that stop a webcam mid-session: sleep/wake, another app taking the camera,
    /// a USB camera being unplugged or re-plugged, media-server errors.
    private func observeSession() {
        let nc = NotificationCenter.default
        nc.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil) { [weak self] n in
            self?.dlog("runtime error \(String(describing: n.userInfo?[AVCaptureSessionErrorKey]))")
            self?.restart(after: 1)
        }
        nc.addObserver(forName: AVCaptureSession.wasInterruptedNotification, object: session, queue: nil) { [weak self] _ in
            DispatchQueue.main.async { self?.onError?("Camera paused — another app may be using it") }
        }
        nc.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil) { [weak self] _ in
            self?.restart(after: 0.2)
        }
        nc.addObserver(forName: AVCaptureDevice.wasDisconnectedNotification, object: nil, queue: nil) { [weak self] n in
            guard let self, let d = n.object as? AVCaptureDevice, d.hasMediaType(.video) else { return }
            if d.uniqueID == self.deviceID { self.restart(after: 1) }   // our camera went away: fall back to another
        }
        nc.addObserver(forName: AVCaptureDevice.wasConnectedNotification, object: nil, queue: nil) { [weak self] n in
            guard let d = n.object as? AVCaptureDevice, d.hasMediaType(.video) else { return }
            self?.restart(after: 1, onlyIfStale: true)                // a camera appeared while we had none
        }
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: nil) { [weak self] _ in
            self?.restart(after: 2, onlyIfStale: true)
        }
    }
    /// Restart the live camera (no-op for replay / synthetic runs). Safe to call repeatedly.
    func restart(after delay: Double, onlyIfStale: Bool = false) {
        guard replayTimer == nil else { return }
        queue.asyncAfter(deadline: .now() + delay) {
            guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
            if onlyIfStale && CACurrentMediaTime() - self.lastFrameAt < 1.5 { return }   // still delivering: leave it alone
            self.dlog("restarting camera")
            if self.session.isRunning { self.session.stopRunning() }
            self.configure()
        }
    }
    /// Time of the last frame from the camera (watchdog for a camera that silently stopped).
    private let frameClock = NSLock()
    private var _lastFrameAt = 0.0
    var lastFrameAt: Double { frameClock.lock(); defer { frameClock.unlock() }; return _lastFrameAt }
    /// While the window is hidden or minimised nothing is processed (recording still gets every frame).
    private var _paused = false
    var paused: Bool {
        get { frameClock.lock(); defer { frameClock.unlock() }; return _paused }
        set { frameClock.lock(); _paused = newValue; frameClock.unlock() }
    }

    var debugCam = 0
    // Frames arrive on `queue`, which only records and hands off; the heavy Vision/bald work runs on `procQueue`
    // with latest-frame semantics. So the recorder gets every frame even while processing is slow (e.g. warm-up).
    let procQueue = DispatchQueue(label: "hairgame.process", qos: .userInteractive)
    private let procLock = NSLock()
    private var procBusy = false
    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameClock.lock(); _lastFrameAt = CACurrentMediaTime(); let p = _paused; frameClock.unlock()
        record(sb)
        if !p { handOff(sb) }
    }
    private func handOff(_ sb: CMSampleBuffer) {
        procLock.lock()
        if procBusy { procLock.unlock(); return }      // still working on the previous frame: skip this one
        procBusy = true
        procLock.unlock()
        procQueue.async {
            self.process(sb)
            self.procLock.lock(); self.procBusy = false; self.procLock.unlock()
        }
    }

    /// Full per-frame pipeline: Vision (pose, landmarks, person mask) → bald overlay → hand the frame to the view.
    func process(_ sb: CMSampleBuffer) {
        debugCam += 1
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        // the person outline is needed every frame: a stale outline makes the head edge jump
        let tProc = CACurrentMediaTime()
        // person outline runs in parallel with the face requests (it is independent of them)
        var frameMask: [UInt8]?
        let segDone = DispatchSemaphore(value: 0)
        segQueue.async {
            Perf.time("v.seg") { try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform([self.segReq]) }
            if let m = self.segReq.results?.first?.pixelBuffer {   // copy out now; Vision recycles its buffers
                var buf = [UInt8](repeating: 0, count: PW * PH)
                let mi = CIImage(cvPixelBuffer: m)
                let mt = mi.transformed(by: CGAffineTransform(scaleX: CGFloat(PW) / mi.extent.width, y: CGFloat(PH) / mi.extent.height))
                self.ci.render(mt, toBitmap: &buf, rowBytes: PW, bounds: CGRect(x: 0, y: 0, width: PW, height: PH), format: .L8, colorSpace: nil)
                frameMask = buf
            }
            segDone.signal()
        }
        Perf.time("v.face") {
            let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
            try? h.perform([rectReq])
            faceReq.inputFaceObservations = rectReq.results           // landmarks reuse the faces just found
            try? h.perform([faceReq])
        }
        if segDone.wait(timeout: .now() + 1) == .timedOut { return }   // never deadlock on a stuck request
        Perf.add("vision", (CACurrentMediaTime() - tProc) * 1000)

        var lm: Landmarks?
        if let f = (faceReq.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width }), let L = f.landmarks {
            let one = CGSize(width: 1, height: 1)
            func pts(_ r: VNFaceLandmarkRegion2D?) -> [CGPoint] { r.map { $0.pointsInImage(imageSize: one) } ?? [] }
            func mean(_ p: [CGPoint]) -> CGPoint? { p.isEmpty ? nil : p.reduce(.zero, +) * (1 / CGFloat(p.count)) }
            let b = f.boundingBox
            // head pose must come from the same person: pick the pose box that overlaps this face the most
            let pose = (rectReq.results ?? []).max { a, c in
                a.boundingBox.intersection(b).width * a.boundingBox.intersection(b).height <
                c.boundingBox.intersection(b).width * c.boundingBox.intersection(b).height
            }.flatMap { $0.boundingBox.intersects(b) ? $0 : nil }
            let ea = mean(pts(L.leftEye)) ?? CGPoint(x: b.minX + 0.3 * b.width, y: b.minY + 0.62 * b.height)
            let eb = mean(pts(L.rightEye)) ?? CGPoint(x: b.minX + 0.7 * b.width, y: b.minY + 0.62 * b.height)
            lm = Landmarks(eyeA: ea, eyeB: eb, brows: pts(L.leftEyebrow) + pts(L.rightEyebrow),
                           contour: pts(L.faceContour), nose: pts(L.nose), lips: pts(L.outerLips),
                           noseTip: pts(L.noseCrest).last,
                           vYaw: pose?.yaw.map { CGFloat($0.doubleValue) },
                           vPitch: pose?.pitch.map { CGFloat($0.doubleValue) })
        }

        // Vision occasionally returns one frame of wildly wrong landmarks (e.g. looking up): the face jumps, shrinks
        // or rolls far more than a head can between frames. Reuse the last good set for up to two such frames
        // (a real fast move persists and is accepted on the next frame).
        if let cur = lm {
            lmMisses = 0
            let mid = (cur.eyeA + cur.eyeB) * 0.5, iod = hypot(cur.eyeA.x - cur.eyeB.x, (cur.eyeA.y - cur.eyeB.y) * 9 / 16)
            let roll = atan2((cur.eyeB.y - cur.eyeA.y) * 9 / 16, cur.eyeB.x - cur.eyeA.x)
            if let g = lastGoodLm, lmRejects < 2 {
                let gm = (g.eyeA + g.eyeB) * 0.5, gi = max(1e-4, hypot(g.eyeA.x - g.eyeB.x, (g.eyeA.y - g.eyeB.y) * 9 / 16))
                let gr = atan2((g.eyeB.y - g.eyeA.y) * 9 / 16, g.eyeB.x - g.eyeA.x)
                var dr = abs(roll - gr); if dr > .pi { dr = 2 * .pi - dr }
                let jump = hypot(mid.x - gm.x, (mid.y - gm.y) * 9 / 16) / gi
                if jump > 0.7 || iod / gi < 0.55 || iod / gi > 1.8 || dr > 0.4 {
                    lmRejects += 1; lm = g
                    dlog(String(format: "landmarks rejected: jump %.2f size %.2f roll %.2f", jump, iod / gi, dr))
                } else { lmRejects = 0; lastGoodLm = cur }
            } else { lmRejects = 0; lastGoodLm = cur }
        } else if let g = lastGoodLm, lmMisses < 2, hasMask {
            // Vision missed the face for a frame (blur, a hand passing): keep the cap on the last landmarks
            // instead of flashing the real hair for one frame
            lmMisses += 1; lm = g
        } else { lastGoodLm = nil; lmRejects = 0 }

        // stage 2 (bald cap) runs on its own queue so the next frame's Vision can start meanwhile.
        // Wait only if the previous frame's cap is still being built (keeps frames in order, never piles up).
        if overlayGate.wait(timeout: .now() + 1) == .timedOut { return }
        overlayQueue.async {
            defer { self.overlayGate.signal() }
            if let m = frameMask { self.maskBuf = m; self.hasMask = true }
            var overlay: CGImage?
            if let lm, self.hasMask {
                overlay = Perf.time("overlay") { self.makeBaldOverlay(CIImage(cvPixelBuffer: pb), lm) }
            }
            self.deliver(sb, pb, overlay, lm, tProc)
        }
    }

    private func deliver(_ sb: CMSampleBuffer, _ pb: CVPixelBuffer, _ overlay: CGImage?, _ lm: Landmarks?, _ tProc: Double) {
        // show this exact frame immediately so the overlay stays in sync with the video
        if let atts = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: true), CFArrayGetCount(atts) > 0 {
            let d = unsafeBitCast(CFArrayGetValueAtIndex(atts, 0), to: CFMutableDictionary.self)
            CFDictionarySetValue(d, Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                                 Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
        }
        let frame = Frame(sample: sb, pixels: pb, overlay: overlay, landmarks: lm)
        Perf.add("proc", (CACurrentMediaTime() - tProc) * 1000)
        DispatchQueue.main.async { self.onFrame?(frame) }
    }

    // MARK: test clips (record a clip once, then tune offline by replaying it)
    //
    // Recording runs on its own queue so it starts the instant it is requested, no matter how busy the camera
    // queue is (e.g. while Vision's models warm up). The camera thread only hands frames over; it never waits.
    private let recQueue = DispatchQueue(label: "hairgame.record", qos: .userInitiated)
    private let recLock = NSLock()
    private var recActive = false          // read from the camera thread, guarded by recLock
    private var recPending = 0             // frames handed over but not yet encoded (guarded by recLock)
    private var writer: AVAssetWriter?     // everything below is only touched on recQueue
    private var writerInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var recFrames = 0, recDropped = 0
    private var recSeconds = 10.0
    private var recStartPTS: CMTime?
    private var recRequested = CACurrentMediaTime()
    private var recLog: URL?
    var onRecordDone: ((URL?) -> Void)?

    /// Recording diagnostics go to a text file next to the clip (the system log hides them).
    private func rlog(_ m: String) {
        guard let url = recLog else { return }
        let line = "\(Date()) [+\(String(format: "%.3f", CACurrentMediaTime() - recRequested))s] \(m)\n"
        if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close() }
        else { try? line.write(to: url, atomically: true, encoding: .utf8) }
    }

    /// Record the raw camera feed (no overlays) for `seconds` to a .mov file.
    func startRecording(to url: URL, seconds: Double = 10) {
        let requested = CACurrentMediaTime()
        recQueue.async {
            guard self.writer == nil else { return }
            self.recRequested = requested
            self.recLog = url.deletingPathExtension().appendingPathExtension("log.txt")
            self.rlog("requested")
            do {
                let w = try AVAssetWriter(outputURL: url, fileType: .mov)
                w.movieFragmentInterval = CMTime(value: 1, timescale: 1)   // flush every second: an early quit still leaves a playable clip
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264,
                                                                                    AVVideoWidthKey: 1280, AVVideoHeightKey: 720])
                input.expectsMediaDataInRealTime = true
                let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: 1280, kCVPixelBufferHeightKey as String: 720])
                guard w.canAdd(input) else { self.rlog("cannot add input"); self.finishRecording(nil); return }
                w.add(input)
                guard w.startWriting() else { self.rlog("startWriting failed: \(String(describing: w.error))"); self.finishRecording(nil); return }
                self.writer = w; self.writerInput = input; self.adaptor = ad
                self.recFrames = 0; self.recDropped = 0; self.recSeconds = seconds; self.recStartPTS = nil
                self.recLock.lock(); self.recActive = true; self.recPending = 0; self.recLock.unlock()
                self.rlog("started \(url.lastPathComponent)")
            } catch {
                self.rlog("writer error \(error)")
                self.finishRecording(nil)
            }
        }
    }

    /// Called from the camera thread for every frame: hands the frame to the recorder without blocking.
    private func record(_ sb: CMSampleBuffer) {
        recLock.lock()
        let active = recActive, busy = recPending >= 4
        if active && !busy { recPending += 1 }
        recLock.unlock()
        guard active, let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        if busy { recQueue.async { self.recDropped += 1 }; return }    // encoder behind: drop rather than stall the camera
        recQueue.async {
            self.append(pb, pts)
            self.recLock.lock(); self.recPending -= 1; self.recLock.unlock()
        }
    }

    private func append(_ src: CVPixelBuffer, _ pts: CMTime) {
        guard let w = writer, let input = writerInput, let ad = adaptor else { return }
        if w.status == .failed { rlog("writer failed: \(String(describing: w.error))"); stopRecording(); return }
        if recStartPTS == nil { recStartPTS = pts; w.startSession(atSourceTime: .zero); rlog("first frame") }
        let t = CMTimeSubtract(pts, recStartPTS!)                     // real capture timing, starting at 0
        guard input.isReadyForMoreMediaData else { recDropped += 1; return }
        var pb: CVPixelBuffer?
        if let pool = ad.pixelBufferPool { CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) }
        if pb == nil { CVPixelBufferCreate(nil, 1280, 720, kCVPixelFormatType_32BGRA, nil, &pb) }
        guard let pb else { return }
        let img = CIImage(cvPixelBuffer: src)
        ci.render(img.transformed(by: CGAffineTransform(scaleX: 1280 / img.extent.width, y: 720 / img.extent.height)), to: pb)
        if ad.append(pb, withPresentationTime: t) {
            recFrames += 1
            if recFrames % 60 == 0 { rlog("frames \(recFrames), dropped \(recDropped)") }
        } else { rlog("append failed: \(String(describing: w.error))") }
        if CMTimeGetSeconds(t) >= recSeconds { stopRecording() }
    }

    /// Finish the clip (also called on quit, so an early stop still leaves a playable file). Runs on recQueue.
    func stopRecording(sync: Bool = false) {
        guard let w = writer, let input = writerInput else { return }
        recLock.lock(); recActive = false; recLock.unlock()
        writer = nil; writerInput = nil; adaptor = nil
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        let frames = recFrames, dropped = recDropped
        w.finishWriting {
            self.rlog("finished status \(w.status.rawValue) frames \(frames) dropped \(dropped) error \(String(describing: w.error))")
            self.finishRecording(w.status == .completed ? w.outputURL : nil)
            sem.signal()
        }
        if sync { _ = sem.wait(timeout: .now() + 8) }
    }
    /// For app quit: finish any recording in progress.
    func stopRecordingForQuit() { recQueue.sync { self.stopRecording(sync: true) } }
    private func finishRecording(_ url: URL?) { DispatchQueue.main.async { self.onRecordDone?(url) } }

    /// Replay a recorded clip (.mov/.mp4) or a folder of images through the full pipeline instead of the camera.
    func startReplay(_ url: URL) {
        if let stall = Env.vars["HAIRGAME_STALLPROC"].flatMap(Double.init) {   // test: slow processing
            procLock.lock(); procBusy = true; procLock.unlock()
            procQueue.async { Thread.sleep(forTimeInterval: stall); self.procLock.lock(); self.procBusy = false; self.procLock.unlock() }
        }
        queue.async {
            var frames: [CVPixelBuffer] = []
            func pixelBuffer(_ img: CIImage) -> CVPixelBuffer? {
                var pb: CVPixelBuffer?
                CVPixelBufferCreate(nil, 1280, 720, kCVPixelFormatType_32BGRA,
                                    [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
                guard let pb else { return nil }
                let scaled = img.transformed(by: CGAffineTransform(scaleX: 1280 / img.extent.width, y: 720 / img.extent.height))
                self.ci.render(scaled, to: pb)
                return pb
            }
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            if isDir.boolValue {
                let files = ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []).sorted()
                    .filter { ["jpg", "jpeg", "png", "heic"].contains(($0 as NSString).pathExtension.lowercased()) }
                for f in files { if let im = CIImage(contentsOf: url.appendingPathComponent(f)), let pb = pixelBuffer(im) { frames.append(pb) } }
            } else {
                let asset = AVURLAsset(url: url)
                let sema = DispatchSemaphore(value: 0)
                var track: AVAssetTrack?
                Task { track = try? await asset.loadTracks(withMediaType: .video).first; sema.signal() }
                sema.wait()
                if let track, let reader = try? AVAssetReader(asset: asset) {
                    let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
                    reader.add(out); reader.startReading()
                    while frames.count < 1800, let sb = out.copyNextSampleBuffer() {
                        if let pb = CMSampleBufferGetImageBuffer(sb), let copy = pixelBuffer(CIImage(cvPixelBuffer: pb)) { frames.append(copy) }
                    }
                }
            }
            guard !frames.isEmpty else { DispatchQueue.main.async { self.onError?("Replay: no frames in \(url.lastPathComponent)") }; return }
            var i = 0
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now(), repeating: 1.0 / 30)
            timer.setEventHandler {
                let pb = frames[i % frames.count]; i += 1
                var fmt: CMVideoFormatDescription?
                CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: pb, formatDescriptionOut: &fmt)
                var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 30),
                                                presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()), decodeTimeStamp: .invalid)
                var sb: CMSampleBuffer?
                if let fmt { CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: pb, formatDescription: fmt, sampleTiming: &timing, sampleBufferOut: &sb) }
                if let sb {
                    self.frameClock.lock(); self._lastFrameAt = CACurrentMediaTime(); self.frameClock.unlock()
                    self.record(sb); self.handOff(sb)
                }
            }
            timer.resume()
            self.replayTimer = timer
        }
    }
    private var replayTimer: DispatchSourceTimer?

    // MARK: self-test (synthetic heads; no camera, no photos)
    /// Renders synthetic scenes (frontal, turned, strongly turned, close, tilted) through the real bald pipeline,
    /// writes before/after images and prints coverage metrics: how much hair remains and how much face was touched.
    func selfTest(_ dir: URL) {
        // geometry checks: world3/local3 must be inverses, unproject must invert world3 on the face surface
        var worst: CGFloat = 0, worstU: CGFloat = 0, worstAt = "", worstR: CGFloat = 0
        let surf = FaceShape().face
        for yawT in [-0.9, -0.4, 0, 0.5, 1.0] as [CGFloat] { for pitchT in [-0.3, 0, 0.3] as [CGFloat] {
            var f = LocalFrame(origin: CGPoint(x: 300, y: 200), scale: 90, roll: 0.2, yaw: yawT, pitch: pitchT)
            f.camDist = 5
            for _ in 0..<50 {
                let p = V3(.random(in: -1.5...1.5), .random(in: -2...2), .random(in: -1...1))
                worst = max(worst, (f.local3(f.world3(p)) - p).len)
                let x = CGFloat.random(in: -0.8...0.8), y = CGFloat.random(in: -1.8 ... -0.2)
                if let q = surf.front(x, y), f.rotate(surf.normal(q)).z > 0.25 {   // only points the camera can see
                    let back = f.unproject(f.world3(q).xy, onto: surf)
                    let err = (back - CGPoint(x: x, y: y)).len
                    let bq = surf.front(back.x, back.y) ?? q
                    let resid = (f.world3(bq).xy - f.world3(q).xy).len / f.scale    // image-space mismatch
                    if abs(yawT) <= 0.6, err > worstU { worstU = err; worstAt = String(format: "yaw %.1f pitch %.1f at (%.2f, %.2f) image residual %.4f", Double(yawT), Double(pitchT), Double(x), Double(y), Double(resid)) }
                    worstR = max(worstR, abs(yawT) <= 0.6 ? resid : 0)
                }
            }
        } }
        print(String(format: "geometry: world3↔local3 max error %.2e   unproject max error %.4f eye-widths ", Double(worst), Double(worstU)) + worstAt + String(format: "  | worst image residual %.4f", Double(worstR)))
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let scenes: [(String, Float, Float, Float, Float)] = [   // name, yaw, iod px, roll, hair volume
            ("frontal", 0, 34, 0, 1), ("turn25", 0.45, 34, 0, 1), ("turn45", 0.8, 34, 0, 1),
            ("turn60_left", -1.05, 34, 0, 1), ("close", 0.2, 58, 0, 1), ("tilted", 0.3, 38, 0.25, 1), ("bighair", 0, 32, 0, 1.8)]
        for (name, yaw, iod, roll, big) in scenes {
            shape = nil; prevAlpha = []; prevScalpW = []; bgCache = []; headMeasure = nil
            for i in 0..<(W0 * H0) { plateW[i] = 0 }
            plateGain = (1, 1, 1)
            var labels = [UInt8](repeating: 0, count: W0 * H0)   // 1 hair, 2 face, 3 other skin
            var lm: Landmarks!
            for _ in 0..<6 {   // a few frames so the background plate and temporal smoothing settle
                lm = synthScene(yaw: yaw, iod: iod, roll: roll, hairScale: big, labels: &labels)
                headYaw = yaw; headPitch = 0; headDist = 1e6; hairVolume = 1
                _ = composeBald(lm)
            }
            let t0 = CACurrentMediaTime()
            for _ in 0..<10 { _ = composeBald(lm) }
            let ms = (CACurrentMediaTime() - t0) * 100
            guard let ov = composeBald(lm) else { continue }
            print(String(format: "%@: bald overlay %.1f ms/frame", name, ms))
            // metrics
            var hairN = 0, hairLeft = 0, faceN = 0, faceTouched = 0
            for i in 0..<(W0 * H0) {
                let a = Float(out[i * 4 + 3]) / 255
                if labels[i] == 1 { hairN += 1; if a < 0.5 { hairLeft += 1 } }
                if labels[i] == 2 { faceN += 1; if a > 0.15 { faceTouched += 1 } }
            }
            print(String(format: "%-12@ hair remaining %5.1f%%   face touched %5.1f%%", name as NSString,
                         100 * Double(hairLeft) / Double(max(1, hairN)), 100 * Double(faceTouched) / Double(max(1, faceN))))
            // before | after image
            let before = CGImage(width: W0, height: H0, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W0 * 4, space: srgb,
                                 bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                 provider: CGDataProvider(data: Data(frameBuf) as CFData)!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
            let ctx = CGContext(data: nil, width: W0 * 2, height: H0, bitsPerComponent: 8, bytesPerRow: 0, space: srgb,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(before, in: CGRect(x: 0, y: 0, width: W0, height: H0))
            ctx.draw(before, in: CGRect(x: W0, y: 0, width: W0, height: H0))
            ctx.draw(ov, in: CGRect(x: W0, y: 0, width: W0, height: H0))
            if let img = ctx.makeImage(), let d = CGImageDestinationCreateWithURL(dir.appendingPathComponent("\(name).png") as CFURL, "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(d, img, nil); CGImageDestinationFinalize(d)
            }
        }
    }
    private var W0: Int { PW }

    /// Synthetic live mode: an animated artificial head (faces forward for calibration, then turns side to side)
    /// driven through the whole app, without the camera.
    func startSynthetic() {
        queue.async {
            let start = CACurrentMediaTime()
            var labels = [UInt8](repeating: 0, count: PW * PH)
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now(), repeating: 1.0 / 30)
            timer.setEventHandler {
                let t = Float(CACurrentMediaTime() - start)
                let yaw: Float = t < 3 ? 0 : 0.85 * sin((t - 3) * 0.7)
                let lm = self.synthScene(yaw: yaw, iod: 36, roll: 0.05 * sin(t * 0.5), hairScale: 1.2, labels: &labels)
                let overlay = self.composeBald(lm)
                var pb: CVPixelBuffer?
                CVPixelBufferCreate(nil, PW, PH, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
                guard let pb else { return }
                CVPixelBufferLockBaseAddress(pb, [])
                let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self), rb = CVPixelBufferGetBytesPerRow(pb)
                for row in 0..<PH { for x in 0..<PW {
                    let i = (row * PW + x) * 4, o = row * rb + x * 4
                    base[o] = self.frameBuf[i + 2]; base[o + 1] = self.frameBuf[i + 1]; base[o + 2] = self.frameBuf[i]; base[o + 3] = 255
                } }
                CVPixelBufferUnlockBaseAddress(pb, [])
                var fmt: CMVideoFormatDescription?
                CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: pb, formatDescriptionOut: &fmt)
                var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 30),
                                                presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()), decodeTimeStamp: .invalid)
                var sb: CMSampleBuffer?
                if let fmt { CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: pb, formatDescription: fmt, sampleTiming: &timing, sampleBufferOut: &sb) }
                guard let sb else { return }
                self.record(sb)
                if let atts = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: true), CFArrayGetCount(atts) > 0 {
                    let d = unsafeBitCast(CFArrayGetValueAtIndex(atts, 0), to: CFMutableDictionary.self)
                    CFDictionarySetValue(d, Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                                         Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
                }
                let frame = Frame(sample: sb, pixels: pb, overlay: overlay, landmarks: lm)
                DispatchQueue.main.async { self.onFrame?(frame) }
            }
            timer.resume()
            self.replayTimer = timer
        }
    }
    private var H0: Int { PH }

    /// Draws a synthetic person into frameBuf/maskBuf and returns its landmarks. Coordinates: y up.
    private func synthScene(yaw: Float, iod: Float, roll: Float, hairScale: Float, labels: inout [UInt8]) -> Landmarks {
        let W = PW, H = PH
        let sy = sin(yaw), cyw = cos(yaw)
        let cx = Float(W) * 0.5, cy = Float(H) * 0.52
        let cr = cos(roll), sr = sin(roll)
        func rot(_ u: Float, _ v: Float) -> (Float, Float) { (cx + (u * cr - v * sr) * iod, cy + (u * sr + v * cr) * iod) }
        func toLocal(_ x: Float, _ y: Float) -> (Float, Float) { let dx = (x - cx) / iod, dy = (y - cy) / iod; return (dx * cr + dy * sr, -dx * sr + dy * cr) }
        // head-space shapes (eye-widths, eye midpoint origin), projected for the turn
        let hairC = (-1.05 * sy, Float(0.65)), hairR = ((1.15 + 0.2 * abs(sy)) * (1 + 0.15 * (hairScale - 1)), 1.6 * (1 + 0.12 * (hairScale - 1)))
        let skullC = (-1.0 * sy, Float(0.55)), skullR = (Float(1.05 + 0.15 * abs(sy)), Float(1.45))
        let faceC = (0.3 * sy, Float(-0.55)), faceR = (Float(0.95 * (0.7 + 0.3 * cyw)), Float(1.35))
        let earC = (-0.98 * sy + (sy >= 0 ? -0.95 : 0.95) * cyw, Float(-0.25))
        func inE(_ u: Float, _ v: Float, _ c: (Float, Float), _ r: (Float, Float)) -> Bool { let a = (u - c.0) / r.0, b = (v - c.1) / r.1; return a * a + b * b < 1 }
        var rng = SystemRandomNumberGenerator()
        for row in 0..<H { for x in 0..<W {
            let i = row * W + x
            let (u, v) = toLocal(Float(x) + 0.5, Float(H - 1 - row) + 0.5)
            var col: (Float, Float, Float) = (0.92 - Float(row) / Float(H) * 0.15, 0.88 - Float(row) / Float(H) * 0.12, 0.8)   // wall
            var m: UInt8 = 0, lab: UInt8 = 0
            let skin: (Float, Float, Float) = (0.87, 0.68, 0.58)
            let hairCol: (Float, Float, Float) = (0.26 + Float.random(in: -0.04...0.04, using: &rng), 0.17, 0.10)
            if v < -2.4 && abs(u - 0.2 * sy) < 2.6 { col = (0.8, 0.12, 0.12); m = 255 }                 // shirt
            else if v < -1.6 && abs(u - 0.2 * sy) < 0.55 { col = skin; m = 255; lab = 3 }               // neck
            if inE(u, v, hairC, hairR) { col = hairCol; m = 255; lab = 1 }
            let earVisible = abs(sy) < 0.2 ? (inE(u, v, (-0.98, -0.25), (0.16, 0.32)) || inE(u, v, (0.98, -0.25), (0.16, 0.32))) : inE(u, v, earC, (0.17, 0.33))
            if earVisible && !inE(u, v, faceC, faceR) { col = (skin.0 * 0.95, skin.1 * 0.92, skin.2 * 0.92); m = 255; lab = 3 }
            if inE(u, v, faceC, faceR) && !(v > 0.95 + 0.35 * (1 - pow(abs(u - faceC.0) / faceR.0, 2)) - 0.2) {   // face below a curved hairline
                col = skin; m = 255; lab = 2
            }
            _ = skullC; _ = skullR
            frameBuf[i * 4] = UInt8(min(255, col.0 * 255)); frameBuf[i * 4 + 1] = UInt8(min(255, col.1 * 255))
            frameBuf[i * 4 + 2] = UInt8(min(255, col.2 * 255)); frameBuf[i * 4 + 3] = 255
            maskBuf[i] = m; labels[i] = lab
        } }
        hasMask = true
        // landmarks (eyes at ±0.5 cos(yaw) around the projected eye midpoint)
        let ex = 0.5 * cyw, shiftN = 0.45 * sy
        func n(_ u: Float, _ v: Float) -> CGPoint { let (x, y) = rot(u, v); return CGPoint(x: CGFloat(x) / CGFloat(W), y: CGFloat(y) / CGFloat(H)) }
        // eyes/brows drawn so the frame looks like a face
        for (u, v, rx, ry, colr) in [(-ex, Float(0), Float(0.16), Float(0.07), (Float(0.2), Float(0.15), Float(0.15))), (ex, 0, 0.16, 0.07, (0.2, 0.15, 0.15)),
                                     (-ex, 0.36, 0.22, 0.05, (0.3, 0.2, 0.12)), (ex, 0.36, 0.22, 0.05, (0.3, 0.2, 0.12)),
                                     (shiftN * 0.6, -1.25, 0.4 * cyw, 0.08, (0.75, 0.38, 0.38))] {
            for row in 0..<H { for x in 0..<W {
                let (lu, lv) = toLocal(Float(x) + 0.5, Float(H - 1 - row) + 0.5)
                if inE(lu, lv, (u, v), (rx, ry)) { let i = row * W + x; frameBuf[i * 4] = UInt8(colr.0 * 255); frameBuf[i * 4 + 1] = UInt8(colr.1 * 255); frameBuf[i * 4 + 2] = UInt8(colr.2 * 255) }
            } }
        }
        let brows = stride(from: Float(-0.75), through: 0.75, by: 0.1).map { n($0 * cyw + 0.1 * sy, 0.38 + 0.05 * (1 - abs($0))) }
        let contour = stride(from: Float(195), through: 345, by: 10).map { a -> CGPoint in
            let t = a * .pi / 180; return n(faceC.0 + faceR.0 * cos(t), faceC.1 + faceR.1 * sin(t)) }
        let nose = [n(shiftN - 0.15, -0.8), n(shiftN, -0.88), n(shiftN + 0.15, -0.8), n(shiftN * 0.5, -0.3)]
        let lips = stride(from: Float(0), through: 330, by: 30).map { a -> CGPoint in
            let t = a * .pi / 180; return n(shiftN * 0.6 + 0.4 * cyw * cos(t), -1.25 + 0.12 * sin(t)) }
        return Landmarks(eyeA: n(-ex, 0), eyeB: n(ex, 0), brows: brows, contour: contour, nose: nose, lips: lips,
                         noseTip: n(shiftN, -0.6), vYaw: CGFloat(yaw), vPitch: 0)
    }

    /// Forget the measured head (outline, face shape, colours) — for Recalibrate. The learned background stays.
    func resetHead() {
        overlayQueue.async {
            self.shape = nil; self.headMeasure = nil; self.dropRatio = nil
            self.prevAlpha = []; self.prevScalpW = []
            self.scalpBase = nil; self.baseRatio = nil; self.camYaw = 0
            self.lastGoodLm = nil; self.lmRejects = 0; self.lmMisses = 0
        }
    }

    /// Full-resolution still of a camera frame (for snapshots).
    func cgImage(_ pb: CVPixelBuffer) -> CGImage? { let i = CIImage(cvPixelBuffer: pb); return ci.createCGImage(i, from: i.extent) }

    // MARK: bald overlay
    //
    // The scalp is derived from the *real* head outline (Vision person mask) rather than a model:
    //   outline ── shrink inward by the hair's thickness ──▶ bald scalp silhouette
    // • hair band (between scalp edge and outline): non-skin pixels → learned background
    // • scalp: repainted as shaded skin (everything above the brows; only hair-coloured pixels lower down,
    //   so ears and temples stay real); fades into the real forehead near the brows
    // • face + brows: protected by a mask built from this frame's landmarks
    // Because the outline turns with the real head, the cap follows any turn, nod or distance.
    var hairVolume: Float = 1.0                                   // hair thickness multiplier (UI)
    /// Grown-hair coverage by direction around the skull (view orientation, 24 bins from -0.3π to 1.3π) and its colour.
    /// Set by the view each frame; the cap paints a dense hair base there, so hair always lines up with the real head.
    // shared between the main thread and the bald-cap queue: always accessed under `sharedLock`
    private let sharedLock = NSLock()
    private var _hairCoverage = [Float](repeating: 0, count: 24)
    private var _hairBase: (Float, Float, Float) = (0.15, 0.1, 0.06)
    private var _headMeasure: (top: CGFloat, halfW: CGFloat)?
    private var _dropRatio: CGFloat?
    /// mouth drop ÷ eye spacing, learned while facing the camera (size cue for strong turns)
    private(set) var dropRatio: CGFloat? {
        get { sharedLock.lock(); defer { sharedLock.unlock() }; return _dropRatio }
        set { sharedLock.lock(); _dropRatio = newValue; sharedLock.unlock() }
    }
    var hairCoverage: [Float] {
        get { sharedLock.lock(); defer { sharedLock.unlock() }; return _hairCoverage }
        set { sharedLock.lock(); _hairCoverage = newValue; sharedLock.unlock() }
    }
    var hairBase: (Float, Float, Float) {
        get { sharedLock.lock(); defer { sharedLock.unlock() }; return _hairBase }
        set { sharedLock.lock(); _hairBase = newValue; sharedLock.unlock() }
    }
    /// Head outline measured while facing the camera (eye-widths from the eye midpoint): top height, half width.
    private(set) var headMeasure: (top: CGFloat, halfW: CGFloat)? {
        get { sharedLock.lock(); defer { sharedLock.unlock() }; return _headMeasure }
        set { sharedLock.lock(); _headMeasure = newValue; sharedLock.unlock() }
    }
    private var dist = [Float](repeating: 0, count: PW * PH)      // px distance to the outline (inside the person)
    private var scalpW = [Float](repeating: 0, count: PW * PH)
    private var prevScalpW: [Float] = []
    private var protWide = [Float](repeating: 0, count: PW * PH)
    private var scalpBase: (Float, Float, Float)?        // forehead colour at the scalp seam
    private var baseRatio: (Float, Float, Float)?        // scalp colour ÷ cheek colour (when the seam is hidden)
    private var sheen: Float = 0.08                      // how shiny the skin is (highlight above typical brightness)
    static let seamBins = 24
    static let bands = max(2, min(8, ProcessInfo.processInfo.activeProcessorCount))
    private var seamSamples = [[(Float, Float, Float)]](repeating: [], count: 24)
    private var seamColor = [(Float, Float, Float)?](repeating: nil, count: 24)   // per strip across the forehead

    private func makeBaldOverlay(_ img: CIImage, _ lm: Landmarks) -> CGImage? {
        let small = img.transformed(by: CGAffineTransform(scaleX: CGFloat(PW) / img.extent.width, y: CGFloat(PH) / img.extent.height))
        ci.render(small, toBitmap: &frameBuf, rowBytes: PW * 4, bounds: CGRect(x: 0, y: 0, width: PW, height: PH), format: .RGBA8, colorSpace: srgb)
        return composeBald(lm)
    }

    /// Smooth value noise in [-1, 1] (head-space coordinates, so it moves with the head).
    @inline(__always) static func valueNoise(_ x: Float, _ y: Float) -> Float {
        let xi = Int32(floor(x)), yi = Int32(floor(y)), fx = x - floor(x), fy = y - floor(y)
        func h(_ a: Int32, _ b: Int32) -> Float { let n = UInt32(bitPattern: a &* 374761393 &+ b &* 668265263) ; let m = (n ^ (n >> 13)) &* 1274126177; return Float(m & 0xffff) / 32767.5 - 1 }
        let sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy)
        let a = h(xi, yi) + (h(xi + 1, yi) - h(xi, yi)) * sx, b = h(xi, yi + 1) + (h(xi + 1, yi + 1) - h(xi, yi + 1)) * sx
        return a + (b - a) * sy
    }
    /// Hair thickness (eye-widths) for a direction from the head centre: thicker on top, thinner at the sides.
    @inline(__always) static func thickness(_ dirUp: Float, _ vol: Float) -> Float { vol * (0.09 + 0.31 * max(0, dirUp) * max(0, dirUp)) }   // short at the sides, thicker on top

    /// Core of the bald effect on frameBuf (RGBA) + maskBuf (person mask), both PW×PH.
    func composeBald(_ lm: Landmarks) -> CGImage? {
        let W = PW, H = PH
        let toPx = { (p: CGPoint) in CGPoint(x: p.x * CGFloat(W), y: p.y * CGFloat(H)) }
        let f = LocalFrame(toPx(lm.eyeA), toPx(lm.eyeB))
        let measured = FaceShape(lm, f, toPx)
        // head pose of *this* frame (Vision's pose for the same image; the view's filtered pose lags on fast turns),
        // lightly smoothed; strong turns corrected with the squeezed eye spacing against the true face size
        let drop = lm.mouthDrop(toPx)
        var rawYaw = lm.vYaw.map { Float($0) } ?? headYaw
        if let d = drop, let R = dropRatio, d > 1 {
            let geo = acos(max(0.2, min(1, Float(f.scale / (d / R)))))
            rawYaw = (rawYaw < 0 ? -1 : 1) * (abs(rawYaw) + max(0, geo - abs(rawYaw)) * Float(dropWeight(CGFloat(rawYaw))))
        }
        camYaw = lm.vYaw == nil ? rawYaw : camYaw + (rawYaw - camYaw) * 0.7
        let pitchNow = lm.vPitch.map { Float($0) } ?? headPitch
        let frontal = abs(camYaw) < 0.25 && abs(pitchNow) < 0.2
        if shape == nil { shape = measured } else if frontal { shape!.blend(measured, 0.2) }
        let sh = shape!

        let yw = max(-1.3, min(1.3, camYaw))
        var iodT = Float(f.scale) / max(0.45, cos(yw))            // true eye spacing in px (eyes look closer when turned)
        // strong turns: Vision squeezes the eyes together far more than cos(turn); the mouth drop keeps the true size
        if frontal, let d = drop, f.scale > 8 { let r = d / f.scale; dropRatio = dropRatio.map { $0 + (r - $0) * 0.1 } ?? r }
        if let d = drop, let R = dropRatio {
            let w = Float(dropWeight(CGFloat(yw)))
            iodT += (max(iodT, Float(d / R)) - iodT) * w
        }
        let ox = Float(f.origin.x), oy = Float(f.origin.y), inv = 1 / iodT
        let c = Float(cos(f.roll)), s = Float(sin(f.roll))
        let browY = Float(sh.browY)
        let u0 = -1.0 * sin(yw)                                     // head bound leans toward the back of the head when turned
        // skull ellipse (eye-widths): from the measured outline when available, widening toward the back when turned
        let hm = headMeasure ?? (top: CGFloat(browY) + 1.9, halfW: 1.2)
        let skullV = browY + 0.1
        let skullRX = (Float(hm.halfW) - Camera.thickness(0, hairVolume) + 0.06) * (1 + 0.5 * abs(sin(yw)))   // a turned head is long front-to-back
        var skullRY = max(1.0, Float(hm.top) - Camera.thickness(1, hairVolume) - skullV)
        let skullU = -1.0 * sin(yw)                                // the skull's centre is well behind the face
        let vol = hairVolume
        // skin and scalp colours are kept exposure-normalised (like the plate) and brought to this frame's exposure
        let eg = plateGain
        let (sr0, sg0, sb0) = (skin.0 * eg.0, skin.1 * eg.1, skin.2 * eg.2)
        let sY = 0.299 * sr0 + 0.587 * sg0 + 0.114 * sb0
        let sCb = sb0 - sY, sCr = sr0 - sY
        let thresh = 0.065 / max(0.3, strength)

        Perf.time("o.prot") { buildProtection(lm, f, toPx, scale: CGFloat(iodT), turn: CGFloat(smoothstep(0.3, 0.8, abs(yw)))) }
        // brow span (eye-widths, same axes as u): above eye level nothing beyond it is face
        let browUs = lm.brows.map { p -> Float in
            let q = toPx(p); let dx = Float(q.x) - ox, dy = Float(q.y) - oy
            return (dx * c + dy * s) * inv
        }
        let browLo = (browUs.min() ?? -0.9) - 0.1, browHi = (browUs.max() ?? 0.9) + 0.1
        let tTemple = CACurrentMediaTime()
        // face outline (jaw contour) in the same units: hair behind it may reach down to the nape
        let jaw = lm.contour.map { p -> (Float, Float) in
            let q = toPx(p); let dx = Float(q.x) - ox, dy = Float(q.y) - oy
            return ((dx * c + dy * s) * inv, (-dx * s + dy * c) * inv)
        }
        // face span (lo, hi) by height, precomputed once per frame (was walked per pixel)
        let spanV0: Float = -2.5, spanStep: Float = 0.02, spanN = 250
        var spanLo = [Float](repeating: .nan, count: spanN), spanHi = [Float](repeating: .nan, count: spanN)
        for k in 0..<spanN {
            let v = spanV0 + Float(k) * spanStep
            var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude, n = 0
            for q in 1..<max(1, jaw.count) {
                let a = jaw[q - 1], b = jaw[q]
                if (a.1 - v) * (b.1 - v) <= 0, abs(b.1 - a.1) > 1e-5 {
                    let x = a.0 + (b.0 - a.0) * (v - a.1) / (b.1 - a.1); lo = min(lo, x); hi = max(hi, x); n += 1
                }
            }
            if n < 2, let f0 = jaw.first, let f1 = jaw.last, v > max(f0.1, f1.1) {   // above the contour's ends: use its top ends
                lo = min(f0.0, f1.0); hi = max(f0.0, f1.0); n = 2
            }
            if n >= 2 { spanLo[k] = lo; spanHi[k] = hi }
        }
        let backDir = -sin(yw)                                       // +: the back of the head is toward +u
        func outsideFace(_ u: Float, _ v: Float) -> Bool { outsideFace(u, v, margin: 0.25) }
        func outsideFace(_ u: Float, _ v: Float, margin: Float) -> Bool {
            let k = Int((v - spanV0) / spanStep)
            guard k >= 0, k < spanN, !spanLo[k].isNaN else { return false }
            let lo = spanLo[k], hi = spanHi[k]
            // only on the back-of-head side when turned (both sides when facing the camera), clear of the cheek edge
            if abs(yw) < 0.2 { return u < lo - margin || u > hi + margin }
            return backDir > 0 ? u > hi + margin : u < lo - margin
        }
        for row in 0..<H { for x in 0..<W {
            let i = row * W + x
            if protMask[i] == 0 { continue }
            let dx = Float(x) + 0.5 - ox, dy = Float(H - 1 - row) + 0.5 - oy
            let u = (dx * c + dy * s) * inv, v = (-dx * s + dy * c) * inv
            if v > -0.1 && (u < browLo || u > browHi) {
                let beyond = u < browLo ? browLo - u : u - browHi            // eye-widths past the brow end
                let fadeUp = min(1, (v + 0.1) / 0.15)                           // full effect from just above eye level
                protMask[i] *= 1 - min(1, beyond / 0.1) * fadeUp
            }
        } }
        Perf.add("o.temple", (CACurrentMediaTime() - tTemple) * 1000)
        let tWide = CACurrentMediaTime()
        protWide = protMask
        let wr = max(2, Int(0.22 * iodT))
        boxBlurR(&protWide, wr); boxBlurR(&protWide, wr)
        Perf.add("o.wide", (CACurrentMediaTime() - tWide) * 1000)
        Perf.time("o.dist") { distanceTransform(smooth: max(3, Int(0.2 * iodT))) }
        // the dome's height comes from this frame's scalp top (the outline shrunk by the hair on top), so it always
        // touches the real top of the head and rounds its corners — whatever the turn, nod or distance
        do {
            // measured along several columns (each scaled up to the dome's crown), median: a single tuft can't lift it
            let thickTop = Camera.thickness(1, vol) * iodT
            var est: [Float] = []
            for k in -2...2 {
                let du = Float(k) * 0.25 * skullRX
                let uw = 0.5 * skullU + du
                var v = skullV + 0.3, top: Float?
                while v < skullV + 3 {
                    let px = ox + (uw * c - v * s) * iodT, py = oy + (uw * s + v * c) * iodT
                    let x = Int(px), row = H - 1 - Int(py)
                    if x < 0 || x >= W || row < 0 || row >= H { break }            // head leaves the frame: no estimate
                    if dist[row * W + x] < thickTop { top = v; break }
                    v += 0.03
                }
                let a = min(0.9, abs(uw - skullU) / skullRX)
                if let t = top { est.append((t - skullV) / powf(1 - powf(a, 2.4), 1 / 2.4)) }
            }
            if est.count >= 3 { skullRY = max(0.6, est.sorted()[est.count / 2]) }
            if overlayFrame % 30 == 0, Env.debug {
                FileHandle.standardError.write("DOME skullV \(skullV) browY \(browY) est \(est.map { (($0 * 100).rounded()) / 100 }) cov \(hairCoverage.map { Int($0 * 9) })\n".data(using: .utf8)!)
            }
        }
        let tMain = CACurrentMediaTime()

        let skinOK = abs(yw) < 0.55
        let turnT = smoothstep(0.3, 0.7, abs(yw))
        let lowLimit: Float = -0.45 - 0.95 * smoothstep(0.2, 0.5, abs(yw))   // how far below the eyes hair may be removed

        // per-pixel pass, split into row bands across cores; each band collects its own colour samples
        let bands = Camera.bands
        var bandFore = [[(Float, Float, Float)]](repeating: [], count: bands), bandCheek = bandFore, bandSeam = bandFore
        var bandBins = [[[(Float, Float, Float)]]](repeating: [[(Float, Float, Float)]](repeating: [], count: Camera.seamBins), count: bands)
        var bandExp = [(Double, Double, Double, Double, Double, Double)](repeating: (0, 0, 0, 0, 0, 0), count: bands)
        let gain = plateGain, ig = (1 / plateGain.0, 1 / plateGain.1, 1 / plateGain.2)
        let bandLock = NSLock()
        frameBuf.withUnsafeBufferPointer { fb in
        maskBuf.withUnsafeBufferPointer { mb in
        plate.withUnsafeMutableBufferPointer { pl in
        plateW.withUnsafeMutableBufferPointer { pw in
        alpha.withUnsafeMutableBufferPointer { al in
        scalpW.withUnsafeMutableBufferPointer { sw in
        protMask.withUnsafeBufferPointer { protMask in
        protWide.withUnsafeBufferPointer { protWide in
        dist.withUnsafeBufferPointer { dist in
        distOut.withUnsafeBufferPointer { distOut in
          DispatchQueue.concurrentPerform(iterations: bands) { band in
            var fore: [(Float, Float, Float)] = [], cheek: [(Float, Float, Float)] = [], seam: [(Float, Float, Float)] = []
            var seamSamples = [[(Float, Float, Float)]](repeating: [], count: Camera.seamBins)
            var exp: (Double, Double, Double, Double, Double, Double) = (0, 0, 0, 0, 0, 0)   // Σ frame, Σ plate·gain on seen background
            let capF = 400 / bands + 1, capC = 600 / bands + 1, capS = 600 / bands + 1, capB = 60 / bands + 2
            for row in (band * H / bands)..<((band + 1) * H / bands) {
                let y = Float(H - 1 - row) + 0.5
                for x in 0..<W {
                    let i = row * W + x
                    let r = Float(fb[i * 4]) / 255, g = Float(fb[i * 4 + 1]) / 255, b = Float(fb[i * 4 + 2]) / 255
                    let pm = Float(mb[i]) / 255
                    let dxh = Float(x) + 0.5 - ox, dyh = y - oy
                    let uh = (dxh * c + dyh * s) * inv, vh = (-dxh * s + dyh * c) * inv
                    let hbu = (uh - u0) / (2.15 + 0.4 * abs(sin(yw))), hbv = (vh - 0.75) / 2.1
                    // a thin ring just outside the head outline (above the ears) often holds light hair the mask missed:
                    // never learn it into the background, and replace it with background
                    // wider on top, where spiky / flyaway tips stick out past the person mask
                    let hl = (uh * uh + (vh - 0.5) * (vh - 0.5)).squareRoot()
                    let haloR = (0.3 + 0.25 * max(0, hl > 0.01 ? (vh - 0.5) / hl : 0)) * iodT
                    let halo = pm < 0.3 && distOut[i] < haloR && vh > -0.2 && hbu * hbu + hbv * hbv < 1
                    if halo {
                        al[i] = smoothstep(haloR, haloR * 0.35, distOut[i]); sw[i] = 0
                        continue
                    }
                    if pm < 0.06 {   // learn the background wherever no person is (exposure-normalised)
                        let nr = r * ig.0, ng = g * ig.1, nb = b * ig.2
                        if pw[i] == 0 { pl[i * 3] = nr; pl[i * 3 + 1] = ng; pl[i * 3 + 2] = nb }
                        else {
                            if (x & 3) == 0 && (row & 3) == 0 {
                                exp.0 += Double(r); exp.1 += Double(g); exp.2 += Double(b)
                                exp.3 += Double(pl[i * 3] * gain.0); exp.4 += Double(pl[i * 3 + 1] * gain.1); exp.5 += Double(pl[i * 3 + 2] * gain.2)
                            }
                            pl[i * 3] += (nr - pl[i * 3]) * 0.25; pl[i * 3 + 1] += (ng - pl[i * 3 + 1]) * 0.25; pl[i * 3 + 2] += (nb - pl[i * 3 + 2]) * 0.25
                        }
                        pw[i] = 1
                    }
                    al[i] = 0; sw[i] = 0
                    if pm < 0.02 { continue }

                    let dx = Float(x) + 0.5 - ox, dy = y - oy
                    let u = (dx * c + dy * s) * inv, v = (-dx * s + dy * c) * inv   // eye-widths, roll removed

                    // skin colour: from the central face only (inside the landmark face mask, between the eyes
                    // and the nose tip, away from the eyes) — never from areas that may be hair when turned
                    if skinOK && pm > 0.6 && (x &+ row) % 2 == 0 && protMask[i] > 0.95 {
                        if v > -0.85 && v < -0.2 && abs(u) < 0.75 { if cheek.count < capC { cheek.append((r, g, b)) } }
                        else if v > browY + 0.1 && v < browY + 0.3 && abs(u) < 0.3 { if fore.count < capF { fore.append((r, g, b)) } }
                    }

                    // only the head: generous bound around the skull (hands, shoulders, phones are left alone)
                    let bu = (u - u0) / (2.15 + 0.4 * abs(sin(yw))), bv = (v - 0.75) / 2.1
                    // below eye level only behind the face outline (back of the head down to the nape when turned)
                    // (only when turned: facing the camera or looking down, beside the jaw is neck and shirt, never hair)
                    guard bu * bu + bv * bv < 1, v > -0.45 || (v > lowLimit && outsideFace(u, v)) else { continue }
                    let prot = protMask[i]
                    if prot > 0.98 { continue }

                    // hair-likeness: far from the skin colour (darker or different chroma)
                    let Y = 0.299 * r + 0.587 * g + 0.114 * b
                    let dY = (Y - sY) * (Y < sY ? 0.9 : 0.45)
                    let dCb = (b - Y - sCb) * 2.2, dCr = (r - Y - sCr) * 2.2
                    let cd = (dY * dY + dCb * dCb + dCr * dCr).squareRoot()
                    // behind the face outline the only skin is the ear: anything not clearly skin there is hair
                    // (brightly lit blond hair is close to skin colour)
                    let behind = v > -0.3 && v < browY + 0.3 && outsideFace(u, v)
                    // the person mask is soft at the outline: treat moderately-inside pixels as fully present,
                    // otherwise the outermost hair (right at the edge) fades out of the cap
                    let pe = smoothstep(0.05, 0.35, pm)
                    let lowBehind = v <= -0.3 && outsideFace(u, v)
                    let chroma = (dCb * dCb + dCr * dCr).squareRoot()
                    let h = pe * (behind ? smoothstep(thresh * 0.45, thresh * 1.05, cd)
                                  : lowBehind ? smoothstep(thresh * 1.2, thresh * 2.2, chroma)
                                  : smoothstep(thresh, thresh * 2.1, cd))

                    // skin right at the seam between real forehead and painted scalp: the scalp colour must match it
                    if h < 0.3 && v > browY + 0.03 && abs(u) < 1.2 && protWide[i] > 0.3 && protWide[i] < 0.9 && prot < 0.95 {
                        let bin = min(Camera.seamBins - 1, max(0, Int((u + 1.2) / 2.4 * Float(Camera.seamBins))))
                        if seamSamples[bin].count < capB { seamSamples[bin].append((r, g, b)) }
                        if seam.count < capS { seam.append((r, g, b)) }
                    }
                    let lu = (u * u + (v - 0.5) * (v - 0.5)).squareRoot()
                    // turned: on the face side the outline below the hairline is forehead skin, not hair
                    let foreheadEdge = (u - skullU) * backDir < 0 ? turnT * (1 - smoothstep(browY + 0.55, browY + 0.9, v)) : 0
                    let thick = Camera.thickness(lu > 0.01 ? (v - 0.5) / lu : 0, vol) * iodT * (1 - foreheadEdge)
                    var inScalp = smoothstep(thick - 1.5, thick + 1.5, dist[i])
                    // round the scalp: also stay inside a skull ellipse fitted to the head (clips boxy hair corners)
                    let eu = (u - skullU) / skullRX, ev = (v - skullV) / skullRY
                    // a bald head is a dome: above the temples the scalp stays inside a rounded skull (super-ellipse),
                    // whatever shape the hair gave the outline; the sides (incl. the back when turned) follow the real outline
                    let topW = smoothstep(skullV + 0.1 * skullRY, skullV + 0.45 * skullRY, v)
                    let se = powf(powf(abs(eu), 2.4) + powf(abs(ev), 2.4), 1 / 2.4)
                    inScalp *= 1 - topW * (1 - smoothstep(1.04, 0.97, se))
                    // above the brows the whole scalp is repainted (no blotchy hairline); below, only hair
                    let upper = smoothstep(browY + 0.1, browY + 0.35, v) * (1 - protWide[i])
                    let scalpA = inScalp * max(h, upper)
                    // between scalp and outline above the brows it is all hair (light hair can look skin-coloured)
                    let bandAll = smoothstep(browY, browY + 0.25, v) * (1 - protWide[i]) * (1 - foreheadEdge)
                    let bandA = (1 - inScalp) * max(h, bandAll * pe)
                    let a = min(1, scalpA + bandA) * (1 - prot)
                    al[i] = a
                    sw[i] = scalpA / max(1e-4, scalpA + bandA)
                }
            }
            bandLock.lock()
            bandFore[band] = fore; bandCheek[band] = cheek; bandSeam[band] = seam; bandBins[band] = seamSamples; bandExp[band] = exp
            bandLock.unlock()
          }
        }}}}}}}}}}
        Perf.add("o.main", (CACurrentMediaTime() - tMain) * 1000)
        let tPost = CACurrentMediaTime()
        let fore = bandFore.flatMap { $0 }, cheek = bandCheek.flatMap { $0 }, seam = bandSeam.flatMap { $0 }
        // exposure drift: how the frame compares with the (gain-scaled) plate on background seen right now
        let ex = bandExp.reduce((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) { ($0.0 + $1.0, $0.1 + $1.1, $0.2 + $1.2, $0.3 + $1.3, $0.4 + $1.4, $0.5 + $1.5) }
        if ex.3 > 200 && ex.4 > 200 && ex.5 > 200 {
            func step(_ g: Float, _ cur: Double, _ pl: Double) -> Float { max(0.4, min(2.5, g * Float(1 + (cur / pl - 1) * 0.5))) }
            plateGain = (step(plateGain.0, ex.0, ex.3), step(plateGain.1, ex.1, ex.4), step(plateGain.2, ex.2, ex.5))
        }
        for k in 0..<Camera.seamBins { seamSamples[k] = bandBins.flatMap { $0[k] } }

        if skinOK { updateSkin((cheek + fore).map { ($0.0 / eg.0, $0.1 / eg.1, $0.2 / eg.2) }) }
        if skinOK && cheek.count > 60 {   // shiny skin gets a matching highlight on the scalp
            let L = cheek.map { 0.299 * $0.0 + 0.587 * $0.1 + 0.114 * $0.2 }.sorted()
            let m = L[L.count / 2], hi = L[L.count * 95 / 100]
            sheen += (max(0, min(0.35, hi - m)) - sheen) * 0.05
        }
        if overlayFrame % 30 == 0, Env.debug {
            FileHandle.standardError.write(String(format: "SKULL yaw %.2f measure %@ rx %.2f ry %.2f u %.2f iodT %.1f\n", yw,
                headMeasure.map { String(format: "(%.2f, %.2f)", Double($0.top), Double($0.halfW)) } ?? "nil", skullRX, skullRY, skullU, iodT).data(using: .utf8)!)
            FileHandle.standardError.write(String(format: "COLOR gain=(%.2f %.2f %.2f) seam=%d fore=%d cheek=%d skin=(%.2f %.2f %.2f) base=%@\n", plateGain.0, plateGain.1, plateGain.2, seam.count, fore.count, cheek.count,
                skin.0, skin.1, skin.2, scalpBase.map { String(format: "(%.2f %.2f %.2f)", $0.0, $0.1, $0.2) } ?? "nil").data(using: .utf8)!)
        }
        // forehead colour at the seam: a bright-ish percentile (shadows from brows/fringe pull a median down)
        func lit(_ a: [(Float, Float, Float)]) -> (Float, Float, Float) {
            let sorted = a.sorted { 0.299 * $0.0 + 0.587 * $0.1 + 0.114 * $0.2 < 0.299 * $1.0 + 0.587 * $1.1 + 0.114 * $1.2 }
            let lo = sorted.count * 40 / 100, hi = max(lo + 1, sorted.count * 70 / 100)   // typical brightness, not highlights
            var acc: (Float, Float, Float) = (0, 0, 0)
            for c in sorted[lo..<hi] { acc.0 += c.0; acc.1 += c.1; acc.2 += c.2 }
            let n = Float(hi - lo); return (acc.0 / n, acc.1 / n, acc.2 / n)
        }
        for k in 0..<Camera.seamBins where seamSamples[k].count >= 6 {
            let m = lit(seamSamples[k])
            if let o = seamColor[k] { seamColor[k] = (o.0 + (m.0 - o.0) * 0.2, o.1 + (m.1 - o.1) * 0.2, o.2 + (m.2 - o.2) * 0.2) } else { seamColor[k] = m }
        }
        if seam.count > 25 {
            let m0 = lit(seam), m = (m0.0 / eg.0, m0.1 / eg.1, m0.2 / eg.2)
            scalpBase = scalpBase == nil ? m : (scalpBase!.0 + (m.0 - scalpBase!.0) * 0.15, scalpBase!.1 + (m.1 - scalpBase!.1) * 0.15, scalpBase!.2 + (m.2 - scalpBase!.2) * 0.15)
        }
        // the seam is hidden (looking up/down, strong turns): the face's lighting still changes, so follow the
        // cheeks with the scalp-to-cheek ratio learned while both were visible
        if let b = scalpBase {
            if seam.count > 25 && skinOK && cheek.count > 30 {
                let r = (b.0 / max(0.05, skin.0), b.1 / max(0.05, skin.1), b.2 / max(0.05, skin.2))
                baseRatio = baseRatio.map { ($0.0 + (r.0 - $0.0) * 0.1, $0.1 + (r.1 - $0.1) * 0.1, $0.2 + (r.2 - $0.2) * 0.1) } ?? r
            } else if seam.count <= 25 && skinOK && cheek.count > 30, let r = baseRatio {
                let t = (skin.0 * r.0, skin.1 * r.1, skin.2 * r.2)
                scalpBase = (b.0 + (t.0 - b.0) * 0.15, b.1 + (t.1 - b.1) * 0.15, b.2 + (t.2 - b.2) * 0.15)
            }
        }
        if frontal { measureHead(f, iodT, browY) }

        if let dumpDir = Env.vars["HAIRGAME_DUMP"], (Env.vars["HAIRGAME_DUMPAT"] ?? "56,60,64").split(separator: ",").contains(Substring(String(overlayFrame))) {
            func dump(_ a: [Float], _ name: String, _ scale: Float) {
                let bytes = a.map { UInt8(max(0, min(255, $0 * scale * 255))) }
                guard let prov = CGDataProvider(data: Data(bytes) as CFData),
                      let img = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: W, space: CGColorSpaceCreateDeviceGray(),
                                        bitmapInfo: CGBitmapInfo(rawValue: 0), provider: prov, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
                      let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: "\(dumpDir)/\(name)-\(overlayFrame).png") as CFURL, "public.png" as CFString, 1, nil) else { return }
                CGImageDestinationAddImage(d, img, nil); CGImageDestinationFinalize(d)
            }
            dump(alpha, "alpha", 1); dump(scalpW, "scalpW", 1); dump(dist, "dist", 1 / 60); dump(protMask, "prot", 1)
            dump(maskBuf.map { Float($0) / 255 }, "mask", 1)
        }
        maxFilter(&alpha); boxBlur(&alpha); boxBlur(&alpha)        // cover hair fringes, soft edge
        boxBlur(&scalpW); boxBlur(&scalpW)
        if prevAlpha.count == alpha.count {                         // light temporal smoothing: no shimmer
            alpha.withUnsafeMutableBufferPointer { A in scalpW.withUnsafeMutableBufferPointer { S in
            prevAlpha.withUnsafeMutableBufferPointer { PA in prevScalpW.withUnsafeMutableBufferPointer { PS in
                for i in 0..<A.count { A[i] = A[i] * 0.65 + PA[i] * 0.35; S[i] = S[i] * 0.65 + PS[i] * 0.35; PA[i] = A[i]; PS[i] = S[i] }
            }}}}
        } else { prevAlpha = alpha; prevScalpW = scalpW }
        overlayFrame += 1
        Perf.add("o.post", (CACurrentMediaTime() - tPost) * 1000)
        if bgCache.isEmpty || overlayFrame % 3 == 0 { bgCache = Perf.time("o.pushpull") { pushPull(plate, plateW, W, H) } }
        let tComp = CACurrentMediaTime()
        let bg = bgCache
        let bgGain = plateGain

        // scalp colour: forehead skin, a touch less saturated
        let gNow = plateGain
        let (k0, k1, k2): (Float, Float, Float) = {
            let (r, g, b) = scalpBase ?? {
                let (r, g, b) = skin; let Y = 0.299 * r + 0.587 * g + 0.114 * b
                return (r + (Y - r) * 0.08, g + (Y - g) * 0.08, b + (Y - b) * 0.08)
            }()
            return (r * gNow.0, g * gNow.1, b * gNow.2)
        }()
        let roundness = 0.9 * iodT                                  // how quickly the scalp curves away at its edge
        let shine = sheen * 0.8
        let coverage = hairCoverage, base = hairBase
        out.withUnsafeMutableBufferPointer { o in
        alpha.withUnsafeBufferPointer { al in
        scalpW.withUnsafeBufferPointer { scalpW in
        dist.withUnsafeBufferPointer { dist in
        bg.withUnsafeBufferPointer { bg in
          DispatchQueue.concurrentPerform(iterations: Camera.bands) { band in
            for row in (band * H / Camera.bands)..<((band + 1) * H / Camera.bands) {
                let y = Float(H - 1 - row) + 0.5
                for x in 0..<W {
                    let i = row * W + x
                    let a = min(1, al[i] * 1.25)
                    if a < 0.01 { o[i * 4] = 0; o[i * 4 + 1] = 0; o[i * 4 + 2] = 0; o[i * 4 + 3] = 0; continue }
                    var cr = bg[i * 3] * bgGain.0, cg = bg[i * 3 + 1] * bgGain.1, cb = bg[i * 3 + 2] * bgGain.2
                    let k = scalpW[i]
                    if k > 0.01 {
                        let dx = Float(x) + 0.5 - ox, dy = y - oy
                        let u = (dx * c + dy * s) * inv, v = (-dx * s + dy * c) * inv
                        let lu = (u * u + (v - 0.5) * (v - 0.5)).squareRoot()
                        let thick = Camera.thickness(lu > 0.01 ? (v - 0.5) / lu : 0, vol) * iodT
                        // shading from distance to the scalp edge: curves away (darker) at the rim, lit from above
                        let t = min(1, max(0, dist[i] - thick) / roundness)
                        let nz = (1 - (1 - t) * (1 - t)).squareRoot()
                        let up = min(1, max(0, (v - browY) / 1.6))
                        // 1.0 where it meets the forehead; gently darker toward the rim, a soft sheen on top
                        let shade = (0.88 + 0.12 * nz) * (0.97 + 0.05 * up)
                        // broad highlight on the crown (lit from above), as strong as the skin's own shine
                        let hu = u - 0.15 * yw, hv = v - (browY + 0.85)
                        let spec = nz * nz * (up * up * 0.04 + shine * expf(-(hu * hu * 1.6 + hv * hv * 2.2)))
                        let b0 = k0, b1 = k1, b2 = k2   // one colour (bright forehead sample): per-strip colours caused stripes
                        let tex = 1 + 0.035 * Camera.valueNoise(u * 3.2, v * 3.2) + 0.015 * Camera.valueNoise(u * 9 + 7, v * 9 + 3)
                        var sr = b0 * shade * tex + spec, sg = b1 * shade * tex + spec, sb = b2 * shade * tex + spec
                        // grown hair: dense hair base over the scalp above the hairline, by direction around the skull
                        let th = atan2(v - (browY + 0.1), -u)              // camera u is mirrored relative to the view
                        let fbin = (th + 0.3 * .pi) / (1.6 * .pi) * 24 - 0.5
                        let b0i = max(0, min(23, Int(fbin.rounded(.down)))), b1i = min(23, b0i + 1)
                        let ft = max(0, min(1, fbin - Float(b0i)))
                        let cov = coverage[b0i] + (coverage[b1i] - coverage[b0i]) * ft
                        if cov > 0.01 {
                            var hlY = browY + 0.8 - 0.5 * min(1, (u / 1.2) * (u / 1.2))
                            if outsideFace(u, v, margin: 0.06) { hlY = min(hlY, browY - 0.95) }   // behind the face hair grows down past the ear top
                            let a = cov * smoothstep(hlY - 0.05, hlY + 0.12, v)
                            // strand texture running outward from the crown (so the base reads as hair, not a cap)
                            let rr = (u * u + (v - browY - 0.1) * (v - browY - 0.1)).squareRoot()
                            let strands = 0.82 + 0.18 * Camera.valueNoise(th * 38, rr * 2.2) + 0.1 * Camera.valueNoise(th * 90, rr * 5)
                            let hs = (0.62 + 0.38 * nz) * strands
                            sr += (base.0 * hs - sr) * a; sg += (base.1 * hs - sg) * a; sb += (base.2 * hs - sb) * a
                        }
                        cr += (sr - cr) * k; cg += (sg - cg) * k; cb += (sb - cb) * k
                    }
                    o[i * 4] = UInt8(max(0, min(1, cr)) * a * 255)
                    o[i * 4 + 1] = UInt8(max(0, min(1, cg)) * a * 255)
                    o[i * 4 + 2] = UInt8(max(0, min(1, cb)) * a * 255)
                    o[i * 4 + 3] = UInt8(a * 255)
                }
            }
          }
        }}}}}
        Perf.add("o.compose", (CACurrentMediaTime() - tComp) * 1000)
        guard let provider = CGDataProvider(data: Data(out) as CFData) else { return nil }
        return CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * 4, space: srgb,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// Chamfer distance (px) from each person pixel to the edge of the person mask. The image border is not an edge.
    private var smoothMask = [Float](repeating: 0, count: PW * PH)
    private var distOut = [Float](repeating: 0, count: PW * PH)   // px from the person outline, outside it
    private func distanceTransform(smooth r: Int) {
        let W = PW, H = PH, big: Float = 1e4
        // smooth the outline first: messy hair makes notches that would otherwise be copied into the scalp edge
        maskBuf.withUnsafeBufferPointer { M in smoothMask.withUnsafeMutableBufferPointer { S in for i in 0..<(W * H) { S[i] = Float(M[i]) / 255 } } }
        Perf.time("d.blur") { boxBlurR(&smoothMask, r); boxBlurR(&smoothMask, r) }
        maskBuf.withUnsafeBufferPointer { M in smoothMask.withUnsafeBufferPointer { S in
        dist.withUnsafeMutableBufferPointer { D in distOut.withUnsafeMutableBufferPointer { O in
            for i in 0..<(W * H) { D[i] = S[i] > 0.5 ? big : 0; O[i] = M[i] > 30 ? 0 : big }
        }}}}
        Perf.time("d.ch1") { chamfer(&dist) }
        Perf.time("d.ch2") { chamfer(&distOut) }
    }

    private func chamfer(_ d: inout [Float]) {
        let W = PW, H = PH, dg: Float = 1.4142
        d.withUnsafeMutableBufferPointer { D in
            for y in 0..<H { let o = y * W
                for x in 0..<W {
                    let i = o + x; var m = D[i]; if m == 0 { continue }
                    if x > 0 { m = min(m, D[i - 1] + 1) }
                    if y > 0 { m = min(m, D[i - W] + 1); if x > 0 { m = min(m, D[i - W - 1] + dg) }; if x < W - 1 { m = min(m, D[i - W + 1] + dg) } }
                    D[i] = m
                } }
            for y in stride(from: H - 1, through: 0, by: -1) { let o = y * W
                for x in stride(from: W - 1, through: 0, by: -1) {
                    let i = o + x; var m = D[i]; if m == 0 { continue }
                    if x < W - 1 { m = min(m, D[i + 1] + 1) }
                    if y < H - 1 { m = min(m, D[i + W] + 1); if x < W - 1 { m = min(m, D[i + W + 1] + dg) }; if x > 0 { m = min(m, D[i + W - 1] + dg) } }
                    D[i] = m
                } }
        }
    }

    /// Measure the head outline (top and width) along the face's axes, while facing the camera.
    private func measureHead(_ f: LocalFrame, _ iodT: Float, _ browY: Float) {
        let W = PW, H = PH
        let ox = Float(f.origin.x), oy = Float(f.origin.y), c = Float(cos(f.roll)), s = Float(sin(f.roll))
        func inside(_ u: Float, _ v: Float) -> Bool? {
            let x = Int(ox + (u * c - v * s) * iodT), yy = Int(oy + (u * s + v * c) * iodT)
            let row = H - 1 - yy
            guard x >= 0, x < W, row >= 0, row < H else { return nil }   // off-screen: unknown
            return maskBuf[row * W + x] > 127
        }
        func walk(_ du: Float, _ dv: Float, from u: Float, _ v: Float) -> Float? {
            var t: Float = 0
            while t < 4 {
                guard let ins = inside(u + du * t, v + dv * t) else { return nil }
                if !ins { return t }
                t += 0.02
            }
            return nil
        }
        guard let top = walk(0, 1, from: 0, browY), let r = walk(1, 0, from: 0, browY + 0.5), let l = walk(-1, 0, from: 0, browY + 0.5) else { return }
        let m = (top: CGFloat(browY + top), halfW: CGFloat((r + l) / 2))
        guard m.top > 0.8, m.top < 3.5, m.halfW > 0.8, m.halfW < 2.5 else { return }
        if let o = headMeasure { headMeasure = (o.top + (m.top - o.top) * 0.1, o.halfW + (m.halfW - o.halfW) * 0.1) } else { headMeasure = m }
    }

    /// Box blur with radius r (separable running sums, cache-friendly, no per-element array checks).
    private func boxBlurR(_ a: inout [Float], _ r: Int) {
        let W = PW, H = PH, inv = 1 / Float(2 * r + 1)
        a.withUnsafeMutableBufferPointer { A in
        tmp.withUnsafeMutableBufferPointer { T in
            for y in 0..<H {                                   // horizontal pass → T
                let o = y * W
                var acc: Float = 0
                for k in -r...r { acc += A[o + min(W - 1, max(0, k))] }
                for x in 0..<W {
                    T[o + x] = acc * inv
                    acc += A[o + min(W - 1, x + r + 1)] - A[o + max(0, x - r)]
                }
            }
            var col = [Float](repeating: 0, count: W)          // vertical pass, row by row with per-column sums
            col.withUnsafeMutableBufferPointer { C in
                for x in 0..<W { C[x] = 0 }
                for k in -r...r { let o = min(H - 1, max(0, k)) * W; for x in 0..<W { C[x] += T[o + x] } }
                for y in 0..<H {
                    let o = y * W, add = min(H - 1, y + r + 1) * W, sub = max(0, y - r) * W
                    for x in 0..<W { A[o + x] = C[x] * inv; C[x] += T[add + x] - T[sub + x] }
                }
            }
        }}
    }

    /// Face + eyebrows region from this frame's landmarks (jawline, temples, raised brows), feathered.
    /// Built from the real landmark positions, so it stays on the face at any turn, nod or distance.
    /// `scale`: true eye spacing in px (Vision squeezes the eyes together on turns); `turn`: 0…1 how far turned
    private func buildProtection(_ lm: Landmarks, _ f: LocalFrame, _ toPx: (CGPoint) -> CGPoint, scale: CGFloat, turn: CGFloat) {
        let W = PW, H = PH
        protBytes.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            ctx.setFillColor(gray: 0, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
            let contour = lm.contour.map(toPx)
            guard contour.count > 4 else { return }
            // brows raised by ~0.22 eye-widths along the face's "up" direction
            let up = CGPoint(x: -sin(f.roll), y: cos(f.roll)) * ((0.1 + 0.04 * turn) * scale)
            let brows = lm.brows.map { toPx($0) + up }
            // order: contour (temple → chin → temple), then brows back across the top
            let a = contour.first!, b = contour.last!
            let dir = (b - a).norm
            let top = brows.sorted { ($0.x - a.x) * dir.x + ($0.y - a.y) * dir.y > ($1.x - a.x) * dir.x + ($1.y - a.y) * dir.y }
            let path = CGMutablePath()
            path.addLines(between: contour + [b + up] + top + [a + up])
            path.closeSubpath()
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.addPath(path); ctx.fillPath()
            // a little margin all round (more when turned: Vision's brows then sit too close to the nose)
            ctx.setStrokeColor(gray: 1, alpha: 1); ctx.setLineWidth((0.08 + 0.12 * turn) * scale); ctx.setLineJoin(.round)
            ctx.addPath(path); ctx.strokePath()
        }
        for i in 0..<(W * H) { protMask[i] = Float(protBytes[i]) / 255 }
        boxBlur(&protMask); boxBlur(&protMask)
    }

    /// Robust (median) skin color, smoothed over time.
    private func updateSkin(_ s: [(Float, Float, Float)]) {
        guard s.count > 30 else { return }
        let r = s.map(\.0).sorted()[s.count / 2], g = s.map(\.1).sorted()[s.count / 2], b = s.map(\.2).sorted()[s.count / 2]
        skin = (skin.0 + (r - skin.0) * 0.2, skin.1 + (g - skin.1) * 0.2, skin.2 + (b - skin.2) * 0.2)
    }

    // separable 3×3 max and box filters (raw pointers)
    private func maxFilter(_ a: inout [Float]) {
        let W = PW, H = PH
        a.withUnsafeMutableBufferPointer { A in tmp.withUnsafeMutableBufferPointer { T in
            for y in 0..<H { let o = y * W
                for x in 0..<W { T[o + x] = max(A[o + max(0, x - 1)], A[o + x], A[o + min(W - 1, x + 1)]) } }
            for y in 0..<H { let u = max(0, y - 1) * W, d = min(H - 1, y + 1) * W, o = y * W
                for x in 0..<W { A[o + x] = max(T[u + x], T[o + x], T[d + x]) } }
        }}
    }
    private func boxBlur(_ a: inout [Float]) {
        let W = PW, H = PH, third: Float = 1.0 / 3
        a.withUnsafeMutableBufferPointer { A in tmp.withUnsafeMutableBufferPointer { T in
            for y in 0..<H { let o = y * W
                for x in 0..<W { T[o + x] = (A[o + max(0, x - 1)] + A[o + x] + A[o + min(W - 1, x + 1)]) * third } }
            for y in 0..<H { let u = max(0, y - 1) * W, d = min(H - 1, y + 1) * W, o = y * W
                for x in 0..<W { A[o + x] = (T[u + x] + T[o + x] + T[d + x]) * third } }
        }}
    }

    /// Push-pull hole filling: fills unknown background from known plate pixels.
    private func pushPull(_ col: [Float], _ w: [Float], _ W: Int, _ H: Int) -> [Float] {
        if W <= 2 || H <= 2 {
            var sum: (Float, Float, Float, Float) = (0, 0, 0, 0)
            for i in 0..<(W * H) { sum.0 += col[i * 3] * w[i]; sum.1 += col[i * 3 + 1] * w[i]; sum.2 += col[i * 3 + 2] * w[i]; sum.3 += w[i] }
            let k = sum.3 > 0 ? 1 / sum.3 : 0
            let fill: [Float] = sum.3 > 0 ? [sum.0 * k, sum.1 * k, sum.2 * k] : [0.3, 0.3, 0.3]
            var o = col
            for i in 0..<(W * H) where w[i] < 1 { for ch in 0..<3 { o[i * 3 + ch] = col[i * 3 + ch] * w[i] + fill[ch] * (1 - w[i]) } }
            return o
        }
        let W2 = (W + 1) / 2, H2 = (H + 1) / 2
        var c2 = [Float](repeating: 0, count: W2 * H2 * 3), w2 = [Float](repeating: 0, count: W2 * H2)
        for y in 0..<H { for x in 0..<W {
            let i = y * W + x, j = (y / 2) * W2 + x / 2, wi = w[i]
            if wi == 0 { continue }
            c2[j * 3] += col[i * 3] * wi; c2[j * 3 + 1] += col[i * 3 + 1] * wi; c2[j * 3 + 2] += col[i * 3 + 2] * wi; w2[j] += wi
        } }
        for j in 0..<(W2 * H2) where w2[j] > 0 {
            let k = 1 / w2[j]; c2[j * 3] *= k; c2[j * 3 + 1] *= k; c2[j * 3 + 2] *= k; w2[j] = min(1, w2[j])
        }
        let coarse = pushPull(c2, w2, W2, H2)
        var o = col
        for y in 0..<H {
            let fy = max(0, min(Float(H2 - 1), (Float(y) + 0.5) / 2 - 0.5))
            let y0 = Int(fy), y1 = min(H2 - 1, y0 + 1), ty = fy - Float(y0)
            for x in 0..<W {
                let i = y * W + x
                if w[i] >= 1 { continue }
                let fx = max(0, min(Float(W2 - 1), (Float(x) + 0.5) / 2 - 0.5))
                let x0 = Int(fx), x1 = min(W2 - 1, x0 + 1), tx = fx - Float(x0)
                for ch in 0..<3 {
                    let a = coarse[(y0 * W2 + x0) * 3 + ch], b = coarse[(y0 * W2 + x1) * 3 + ch]
                    let cc = coarse[(y1 * W2 + x0) * 3 + ch], d = coarse[(y1 * W2 + x1) * 3 + ch]
                    let top = a + (b - a) * tx, bot = cc + (d - cc) * tx
                    o[i * 3 + ch] = col[i * 3 + ch] * w[i] + (top + (bot - top) * ty) * (1 - w[i])
                }
            }
        }
        return o
    }
}

// MARK: - Hair model

enum StrandKind { case scalp, beard, mustache }

enum Tool: Int, CaseIterable { case brush, serum, razor, scissors, dye, gel, curl, blow
    var icon: String { ["🪮", "🧪", "🪒", "✂️", "🎨", "💧", "➰", "💨"][rawValue] }
    var name: String { ["Brush", "Serum", "Razor", "Scissors", "Dye", "Gel", "Curler", "Blow"][rawValue] }
    var hint: String {
        ["Drag through hair to comb & style it (styles stick)",
         "Hold on your scalp, jaw or lip to grow hair",
         "Hold on hair to shave it to the skin",
         "Drag across hair to trim it",
         "Hold on hair to dye it the selected color",
         "Hold to stiffen hair, then drag to sculpt it",
         "Hold on hair to curl it (brush straightens)",
         "Hold to blow-dry — air pushes hair around"][rawValue]
    }
}

/// A simulated guide hair; it is drawn as a small clump of child hairs.
final class Strand {
    let kind: StrandKind
    let root: V3               // face-local position on the head surface
    let baseDir: V3
    let maxLen: CGFloat
    let baseStiffness: CGFloat
    var dir: V3                // styled growth direction (face-local)
    var length: CGFloat = 0
    var stiffness: CGFloat
    var curl: CGFloat = 0
    var bend: CGFloat = 0
    let lenVar = CGFloat.random(in: 0.78...1.08)
    var lipT: CGFloat?          // mustache: height above the live upper lip (0 = lip edge, 1 = nose)
    var jaw: (t: CGFloat, inset: CGFloat)?   // beard: position relative to the live jawline
    var skullQ: V3?             // scalp: position on the unit skull (follows skull refits)
    var color = RGB(0.2, 0.12, 0.06)
    let width: CGFloat
    let children: [(offset: CGFloat, shade: CGFloat, phase: CGFloat)]
    var pts: [V3]
    var prev: [V3]
    var n: Int { pts.count - 1 }

    init(_ kind: StrandKind, root: V3, dir: V3, maxLen: CGFloat, segs: Int, stiffness: CGFloat, width: CGFloat, children: Int, spread: CGFloat) {
        self.kind = kind; self.root = root; self.dir = dir.norm; baseDir = dir.norm
        self.maxLen = maxLen; self.stiffness = stiffness; baseStiffness = stiffness; self.width = width
        self.children = (0..<children).map { _ in (CGFloat.random(in: -spread...spread), CGFloat.random(in: 0.72...1.2), CGFloat.random(in: 0...(2 * .pi))) }
        pts = Array(repeating: .zero, count: segs + 1); prev = pts
    }
    func resetStyle() { dir = baseDir; stiffness = baseStiffness; curl = 0; bend = 0 }
}

struct Particle { var p: CGPoint; var v: CGPoint; var color: RGB; var life: CGFloat; var size: CGFloat
    var angle: CGFloat = 0; var spin: CGFloat = 0; var isHair = false }

/// One vertex of a hair ribbon: position (points), signed edge distance + half width (pixels), color.
struct Vtx { var x, y, ex, hw: Float; var rgba: UInt32 }
@inline(__always) func pack(_ r: Float, _ g: Float, _ b: Float, _ a: Float) -> UInt32 {
    func c(_ v: Float) -> UInt32 { UInt32(max(0, min(1, v)) * 255 + 0.5) }
    return c(r) | c(g) << 8 | c(b) << 16 | c(a) << 24
}

/// Builds anti-aliased, tapered ribbons for polylines into one triangle strip (raw buffer, no per-vertex overhead).
final class StrokeBuilder {
    private(set) var ptr: UnsafeMutablePointer<Vtx>
    private(set) var count = 0
    private var cap = 1 << 18
    var px: Float = 2                       // device pixels per point
    init() { ptr = .allocate(capacity: cap) }
    func reset(_ scale: CGFloat) { count = 0; px = Float(scale) }
    @inline(__always) private func reserve(_ n: Int) {
        guard count + n > cap else { return }
        var c = cap; while c < count + n { c *= 2 }
        let np = UnsafeMutablePointer<Vtx>.allocate(capacity: c)
        np.update(from: ptr, count: count); ptr.deallocate(); ptr = np; cap = c
    }
    @inline(__always) private func pair(_ x: Float, _ y: Float, _ nx: Float, _ ny: Float, _ hw: Float, _ c: UInt32, first: Bool) {
        let ext = max(hw, 0.5) + 1, o = ext / px
        let va = Vtx(x: x + nx * o, y: y + ny * o, ex: -ext, hw: hw, rgba: c)
        if first && count > 0 { ptr[count] = ptr[count - 1]; ptr[count + 1] = va; count += 2 }   // degenerate join
        ptr[count] = va
        ptr[count + 1] = Vtx(x: x - nx * o, y: y - ny * o, ex: ext, hw: hw, rgba: c)
        count += 2
    }
    /// General polyline with a per-point color function (used for shadows and particles).
    func add(_ p: UnsafeBufferPointer<CGPoint>, width: CGFloat, taper: CGFloat, color: (Int, Float) -> (Float, Float, Float, Float)) {
        let m = p.count
        guard m >= 2 else { return }
        reserve(m * 2 + 2)
        let w0 = Float(width) * px
        for j in 0..<m {
            let t = Float(j) / Float(m - 1)
            let a = p[max(0, j - 1)], b = p[min(m - 1, j + 1)]
            var tx = Float(b.x - a.x), ty = Float(b.y - a.y)
            let l = (tx * tx + ty * ty).squareRoot()
            if l > 1e-5 { tx /= l; ty /= l } else { tx = 1; ty = 0 }
            let (r, g, bl, al) = color(j, t)
            pair(Float(p[j].x), Float(p[j].y), ty, -tx, w0 * (1 - Float(taper) * t) * 0.5, pack(r, g, bl, al), first: j == 0)
        }
    }
    /// Fast path for hair: positions + precomputed normals, lighting and alpha per point.
    func ribbon(_ a: Int, _ b: Int, m: Int, pts: UnsafePointer<CGPoint>, nrm: UnsafePointer<CGPoint>,
                lum: UnsafePointer<Float>, spec: UnsafePointer<Float>, alpha: UnsafePointer<Float>,
                rgb: (Float, Float, Float), shine: Float, width: CGFloat, taper: Float) {
        guard b - a >= 2 else { return }
        reserve((b - a) * 2 + 2)
        let w0 = Float(width) * px * 0.5, inv = 1 / Float(m - 1)
        for j in a..<b {
            let t = Float(j) * inv
            let sp = spec[j] * shine, l = lum[j]
            // sheen brightens the hair's own colour (adding white washed curly hair out to grey)
            let k = l * (1 + 1.4 * sp)
            let c = pack(rgb.0 * k + sp * 0.08, rgb.1 * k + sp * 0.07, rgb.2 * k + sp * 0.06, alpha[j])
            pair(Float(pts[j].x), Float(pts[j].y), Float(nrm[j].x), Float(nrm[j].y), w0 * (1 - taper * t), c, first: j == a)
        }
    }
}

/// Draws hair ribbons on the GPU (one draw call) into a CAMetalLayer or an offscreen texture.
final class HairRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    let layer = CAMetalLayer()
    private var buffers: [MTLBuffer] = []
    private var bufIndex = 0
    /// At most 3 frames in flight: a buffer is never rewritten while the GPU may still be reading it.
    private let inFlight = DispatchSemaphore(value: 3)
    private func buffer(_ verts: UnsafeMutablePointer<Vtx>, _ count: Int) -> MTLBuffer? {
        let len = max(1, count) * MemoryLayout<Vtx>.stride
        if buffers.count < 3 || buffers[bufIndex].length < len {
            guard let b = device.makeBuffer(length: max(len * 3 / 2, 1 << 20), options: .storageModeShared) else { return nil }
            if buffers.count < 3 { buffers.append(b); bufIndex = buffers.count - 1 } else { buffers[bufIndex] = b }
        }
        let b = buffers[bufIndex]
        bufIndex = (bufIndex + 1) % 3
        b.contents().copyMemory(from: verts, byteCount: count * MemoryLayout<Vtx>.stride)
        return b
    }

    init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        device = d; queue = q
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        struct V { packed_float2 pos; float ex; float hw; uchar4 color; };
        struct O { float4 pos [[position]]; float ex; float hw; float4 color; };
        struct U { float2 size; float2 offset; float k; };
        vertex O vmain(const device V* v [[buffer(0)]], constant U& u [[buffer(1)]], uint id [[vertex_id]]) {
            O o; float2 p = (v[id].pos - u.offset) * u.k / u.size * 2.0 - 1.0;
            o.pos = float4(p, 0, 1); o.ex = v[id].ex; o.hw = v[id].hw; o.color = float4(v[id].color) / 255.0; return o;
        }
        fragment float4 fmain(O i [[stage_in]]) {
            float cov = saturate(max(i.hw, 0.5) + 0.5 - abs(i.ex)) * saturate(i.hw * 2.0);
            float a = i.color.a * cov;
            return float4(i.color.rgb * a, a);
        }
        """
        guard let lib = try? d.makeLibrary(source: src, options: nil) else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "vmain")
        desc.fragmentFunction = lib.makeFunction(name: "fmain")
        let ca = desc.colorAttachments[0]!
        ca.pixelFormat = .bgra8Unorm
        ca.isBlendingEnabled = true
        ca.sourceRGBBlendFactor = .one; ca.sourceAlphaBlendFactor = .one
        ca.destinationRGBBlendFactor = .oneMinusSourceAlpha; ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        guard let p = try? d.makeRenderPipelineState(descriptor: desc) else { return nil }
        pipeline = p
        layer.device = d
        layer.pixelFormat = .bgra8Unorm
        layer.isOpaque = false
        layer.framebufferOnly = true
        layer.maximumDrawableCount = 3
        layer.allowsNextDrawableTimeout = true
        layer.actions = ["bounds": NSNull(), "position": NSNull(), "contents": NSNull()]
    }

    private func encode(_ cb: MTLCommandBuffer, _ tex: MTLTexture, _ verts: UnsafeMutablePointer<Vtx>, _ count: Int, size: CGSize, offset: CGPoint, k: CGFloat, own: Bool = false) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = tex
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rp.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        // snapshots get their own buffer (never one of the display ring's, which may be in flight)
        let vb = own ? device.makeBuffer(bytes: verts, length: max(1, count) * MemoryLayout<Vtx>.stride, options: .storageModeShared) : buffer(verts, count)
        if count >= 3, let buf = vb {
            var u: [Float] = [Float(size.width), Float(size.height), Float(offset.x), Float(offset.y), Float(k), 0]
            enc.setRenderPipelineState(pipeline)
            enc.setVertexBuffer(buf, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: u.count * 4, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: count)
        }
        enc.endEncoding()
    }

    func draw(_ verts: UnsafeMutablePointer<Vtx>, _ count: Int, viewSize: CGSize, scale: CGFloat) {
        let ds = CGSize(width: viewSize.width * scale, height: viewSize.height * scale)
        if layer.drawableSize != ds { layer.drawableSize = ds }
        guard ds.width > 0 else { return }
        guard inFlight.wait(timeout: .now()) == .success else { return }   // GPU behind: skip this frame, never stall
        guard let drawable = layer.nextDrawable(), let cb = queue.makeCommandBuffer() else { inFlight.signal(); return }
        encode(cb, drawable.texture, verts, count, size: viewSize, offset: .zero, k: 1)
        cb.addCompletedHandler { [inFlight] _ in inFlight.signal() }
        cb.present(drawable)
        cb.commit()
    }

    /// Offscreen render for snapshots: maps the view-space `rect` onto a W×H image.
    func image(_ verts: UnsafeMutablePointer<Vtx>, _ count: Int, rect: CGRect, W: Int, H: Int) -> CGImage? {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: W, height: H, mipmapped: false)
        td.usage = [.renderTarget]; td.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: td), let cb = queue.makeCommandBuffer() else { return nil }
        encode(cb, tex, verts, count, size: CGSize(width: W, height: H), offset: rect.origin, k: CGFloat(W) / rect.width, own: true)
        cb.commit(); cb.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: W * H * 4)
        tex.getBytes(&bytes, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
        guard let prov = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                       provider: prov, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

// MARK: - Game view

final class HairView: NSView {
    var camera: Camera?
    var lastPixels: CVPixelBuffer?
    var overlayImage: CGImage?
    var landmarks: Landmarks?
    var lastFaceTime = 0.0
    var imageSize = CGSize(width: 1280, height: 720)
    var face: LocalFrame?
    var shape = FaceShape()
    var live = FaceShape()          // this frame's landmarks (lips/jaw move when you talk or smile)
    private var pitchSamples: [CGFloat] = []      // nose-tip heights while calibrating
    private var pitchBias: CGFloat = 0
    private let pitchSign: CGFloat = CGFloat(Double(Env.vars["HAIRGAME_PITCHSIGN"] ?? "1") ?? 1)
    private var yaw: CGFloat = 0, pitch: CGFloat = 0
    private var debugCue = "-"
    private var debugIod: CGFloat = 0, debugK: CGFloat = 0
    private var camDistance: CGFloat = 0
    private var lostFrame: LocalFrame?
    private var smoothed: LocalFrame?
    private var filteredAt: Double = 0
    private var fOx = OneEuro(1.2, 0.8), fOy = OneEuro(1.2, 0.8)       // origin (eye-widths)
    private var fScale = OneEuro(0.8, 0.02), fRoll = OneEuro(1.0, 0.4)
    private var fYaw = OneEuro(1.0, 0.6), fPitch = OneEuro(0.8, 0.6)
    private var poseOutliers = 0
    var calibration = 0
    let calibrationFrames = 20
    var strands: [Strand] = []
    var particles: [Particle] = []
    var tool: Tool = .serum
    var radius: CGFloat = 40
    var color = RGB(0.24, 0.14, 0.07)
    var bald = true { didSet { overlayLayer.isHidden = !bald } }
    var status = "Waiting for camera…"
    private var calibrationHint: String?
    private let startedAt = CACurrentMediaTime()
    var toast: (String, Double)?
    var onToolChanged: ((Tool) -> Void)?
    var onBaldChanged: ((Bool) -> Void)?
    var onRadiusChanged: ((CGFloat) -> Void)?

    private let root = CALayer()
    private let videoLayer = AVSampleBufferDisplayLayer()
    private let overlayLayer = CALayer()
    private let gpu = HairRenderer()
    private var hairLayer: CALayer { gpu?.layer ?? fallbackLayer }
    private let fallbackLayer = CALayer()
    private let builder = StrokeBuilder()
    private var drawOrder: [Strand] = []
    private var cp = [CGPoint](repeating: .zero, count: 64)
    private var pts2 = [CGPoint](repeating: .zero, count: 64)
    private var vis = [Bool](repeating: true, count: 64)
    private var nrm = [CGPoint](repeating: .zero, count: 64)
    private var lum = [Float](repeating: 0, count: 64)
    private var spec = [Float](repeating: 0, count: 64)
    private var alph = [Float](repeating: 0, count: 64)
    private var fadeTmp = [Float](repeating: 0, count: 64)
    private var runs: [(Int, Int)] = []
    private var depthKey: [CGFloat] = []
    private var taperT = [CGFloat](repeating: 0, count: 64)
    private let cursorLayer = CAShapeLayer()
    private let iconLayer = CATextLayer()
    private let hudLayer = CATextLayer()
    private let keysLayer = CATextLayer()

    private var mouse: CGPoint?
    private var mouseIsDown = false
    private var dragDelta = CGPoint.zero
    private var timer: Timer?
    private var targets = [V3](repeating: .zero, count: 32)

    override init(frame: NSRect) {
        super.init(frame: frame)
        layer = root
        wantsLayer = true
        root.backgroundColor = CGColor(gray: 0.08, alpha: 1)
        root.actions = ["sublayers": NSNull()]
        videoLayer.videoGravity = .resize
        overlayLayer.contentsGravity = .resize
        overlayLayer.minificationFilter = .linear; overlayLayer.magnificationFilter = .linear
        for l in [videoLayer, overlayLayer] as [CALayer] {
            l.actions = ["bounds": NSNull(), "position": NSNull(), "contents": NSNull(), "transform": NSNull(), "hidden": NSNull()]
            l.transform = CATransform3DMakeScale(-1, 1, 1)   // mirror like a real mirror
        }
        cursorLayer.fillColor = nil; cursorLayer.strokeColor = CGColor(gray: 1, alpha: 0.8); cursorLayer.lineWidth = 1.5
        cursorLayer.actions = ["path": NSNull(), "lineDashPattern": NSNull()]
        for t in [iconLayer, hudLayer, keysLayer] {
            t.actions = ["string": NSNull(), "position": NSNull(), "bounds": NSNull(), "hidden": NSNull()]
            t.foregroundColor = .white
            t.anchorPoint = .zero
        }
        iconLayer.fontSize = 22; iconLayer.bounds = CGRect(x: 0, y: 0, width: 40, height: 30)
        hudLayer.font = NSFont.systemFont(ofSize: 15, weight: .semibold); hudLayer.fontSize = 15
        hudLayer.backgroundColor = CGColor(gray: 0, alpha: 0.45); hudLayer.cornerRadius = 6
        keysLayer.fontSize = 12; keysLayer.backgroundColor = CGColor(gray: 0, alpha: 0.35); keysLayer.cornerRadius = 5
        keysLayer.string = " 1–8 tools · B bald · scroll = brush size · space = snapshot "
        for l in [videoLayer, overlayLayer, hairLayer, cursorLayer, iconLayer, hudLayer, keysLayer] as [CALayer] { root.addSublayer(l) }

        timer = Timer(timeInterval: 1.0 / 60, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(timer!, forMode: .common)
    }
    required init?(coder: NSCoder) { fatalError() }
    override var acceptsFirstResponder: Bool { true }

    override func viewDidChangeBackingProperties() {
        let s = window?.backingScaleFactor ?? 2
        for t in [iconLayer, hudLayer, keysLayer] { t.contentsScale = s }
        hairLayer.contentsScale = s
    }

    var calibrated: Bool { calibration >= calibrationFrames }
    func recalibrate() {
        pitchSamples.removeAll(); lostFrame = nil; calibration = 0; strands.removeAll(); drawOrder.removeAll(); particles.removeAll()
        smoothed = nil; filteredAt = 0; camDistance = 0; harnessApplied = false
        shape = FaceShape(); live = FaceShape()
        camera?.resetHead()
        toast = ("🔄 Recalibrating — look straight at the camera", CACurrentMediaTime() + 2.5)
    }

    // MARK: roots
    private func makeStrands() {
        strands.removeAll()
        let sk = shape.skull, fc = shape.face
        // scalp: on the skull ellipsoid above a curved hairline (front, sides and back)
        var count = 0, tries = 0
        while count < 750 && tries < 100000 {
            tries += 1
            let x = CGFloat.random(in: -sk.r.x...sk.r.x), y = CGFloat.random(in: (shape.browY - 0.2)...(sk.c.y + sk.r.y))
            guard var p = sk.front(x, y) else { continue }
            let temple = abs(x) > sk.r.x * 0.82 && y > shape.browY - 0.15
            guard y > shape.hairline(x) || temple else { continue }
            if Bool.random() && abs(x) > sk.r.x * 0.3 { p.z = sk.c.z - (p.z - sk.c.z) * CGFloat.random(in: 0...0.8) }   // back of the head
            let n = sk.normal(p)
            let side: CGFloat = x >= 0 ? 1 : -1
            let d = n * 0.45 + V3(side * 0.45, 0.12, -0.75)   // hair parts in the middle and flows back
            let st = Strand(.scalp, root: p, dir: d, maxLen: 5.0, segs: 14, stiffness: 0.22, width: 0.9, children: 4, spread: 0.08)
            st.skullQ = sk.q(p)
            strands.append(st)
            count += 1
        }
        // beard: anchored to the jawline landmarks (so it follows your real jaw when you turn or talk)
        count = 0; tries = 0
        while count < 420 && tries < 100000 {
            tries += 1
            let t = CGFloat.random(in: 0...1)
            let inset: CGFloat = pow(CGFloat.random(in: 0...1), 1.4) * 0.6   // denser along the jaw edge
            guard let q = shape.beardPoint(t, inset) else { break }
            let x = q.x, y = q.y
            // sideburns: a thin band along the jaw edge only, up to the same height on both sides
            let side = abs(x) > shape.halfWidth * 0.75 && inset < 0.15
            guard y < shape.noseY - 0.12 || (side && y < shape.noseY * 0.35) else { continue }
            guard !(abs(x) < shape.mouthHalf + 0.08 && y > shape.lipBottom - 0.1 && y < shape.lipTop + 0.1) else { continue }
            guard !(y > shape.lipTop - 0.05 && abs(x) < shape.halfWidth * 0.6) else { continue }
            let p = V3(x, y, fc.front(x, y)?.z ?? fc.c.z)
            let n = fc.normal(p)
            let st = Strand(.beard, root: p, dir: n * 0.3 + V3(-p.x * 0.25, -1, 0.2), maxLen: 3.2, segs: 9, stiffness: 0.3, width: 0.9, children: 4, spread: 0.05)
            st.jaw = (t, inset)
            strands.append(st)
            count += 1
        }
        // mustache: between nose and upper lip
        // roots sit just above the upper lip and follow its curve, filling the lower part of the
        // space up toward the nose (never on the nose itself)
        for _ in 0..<160 {
            let x = CGFloat.random(in: -(shape.mouthHalf + 0.03)...(shape.mouthHalf + 0.03))
            let lip = shape.upperLipY(x) + 0.025
            let room = max(0.03, shape.noseY - 0.07 - lip)
            let t = pow(CGFloat.random(in: 0...1), 1.6) * 0.6             // denser near the lip
            let y = lip + room * t
            guard let p = fc.front(x, y) else { continue }
            let st = Strand(.mustache, root: p, dir: V3(x * 1.2, -1, 0.45), maxLen: 1.1, segs: 7, stiffness: 0.4, width: 1.0, children: 3, spread: 0.035)
            st.lipT = t
            strands.append(st)
        }
        for s in strands { s.color = color }
        drawOrder = strands.sorted { $0.root.z < $1.root.z }   // back of head first
    }

    // MARK: frames
    var debugRecv = 0
    func receive(_ fr: Frame) {
        Perf.add("latency", (CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock())) - CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(fr.sample))) * 1000)
        debugRecv += 1
        lastPixels = fr.pixels
        let sz = CGSize(width: CVPixelBufferGetWidth(fr.pixels), height: CVPixelBufferGetHeight(fr.pixels))
        if sz != imageSize { imageSize = sz; needsLayout = true }
        let r = videoLayer.sampleBufferRenderer
        if r.status == .failed { r.flush() }
        r.enqueue(fr.sample)
        overlayImage = fr.overlay
        overlayLayer.contents = fr.overlay
        if let lm = fr.landmarks { landmarks = lm; lastFaceTime = CACurrentMediaTime() }
    }

    // MARK: actions
    func shaveAll() { for s in strands where s.length > 0 { clip(s, from: 0) } }
    func resetStyle() { strands.forEach { $0.resetStyle() } }

    func applyPreset(_ name: String) {
        guard calibrated, let f = face else { notReady(); return }
        let sk = shape.skull
        func set(_ s: Strand, _ len: CGFloat, dir: V3? = nil, stiff: CGFloat? = nil, curl: CGFloat = 0, bend: CGFloat = 0) {
            if len <= 0 { if s.length > 0 { clip(s, from: 0) }; return }
            let fresh = s.length == 0
            if fresh { seed(s, f) }
            s.length = min(s.maxLen, len * (len > 0.3 ? s.lenVar : 1))
            s.dir = (dir ?? s.baseDir).norm; s.stiffness = stiff ?? s.baseStiffness; s.curl = curl; s.bend = bend
            if fresh {   // start laid out along the style (springing out from the root caused a burst)
                let root = rootWorld(s, f), d = f.worldDir(s.dir), seg = s.length * f.scale / CGFloat(s.n)
                for i in 0...s.n { s.pts[i] = root + d * (seg * CGFloat(i)); s.prev[i] = s.pts[i] }
            }
        }
        for s in strands {
            let r = s.root
            let n = s.kind == .scalp ? sk.normal(r) : shape.face.normal(r)
            let front = r.z > sk.c.z + sk.r.z * 0.55
            switch (s.kind, name) {
            case (.scalp, "Bald"): set(s, 0)
            case (.scalp, "Buzz Cut"): set(s, 0.08, dir: n, stiff: 0.9)
            case (.scalp, "Crew Cut"): set(s, 0.3, dir: n * 0.35 + V3(0, 0.45, -0.65), stiff: 0.6)   // short, lying back over the head
            case (.scalp, "Bob"):
                if front && abs(r.x) < 0.8 { set(s, max(0.25, (r.y - shape.browY) * 1.15), dir: V3(r.x * 0.3, -0.7, 0.7), stiff: 0.3) }
                else { set(s, max(0.6, r.y - shape.chinY - 0.2), dir: n * 0.5 + V3(0, -0.6, 0), stiff: 0.18) }
            case (.scalp, "Long"): set(s, 4.6, dir: n * 0.4 + V3((r.x >= 0 ? 1 : -1) * 0.7, 0.1, -0.55), stiff: 0.16)
            case (.scalp, "Slicked Back"): set(s, 1.6, dir: n * 0.25 + V3(0, 0.2, -1), stiff: 0.55)
            case (.scalp, "Side Part"):
                let right = r.x > -0.35
                set(s, 1.3, dir: n * 0.25 + V3(right ? 0.9 : -0.9, -0.15, -0.3), stiff: 0.4)
            case (.scalp, "Mohawk"):
                if abs(r.x) < 0.24 { set(s, 1.5, dir: n * 0.4 + V3(0, 1, -0.15), stiff: 0.92) } else { set(s, 0) }
            case (.scalp, "Spiky"): set(s, 0.8 + .random(in: -0.15...0.15), dir: n + V3(.random(in: -0.3...0.3), 0.3, .random(in: -0.3...0.3)), stiff: 0.92)
            case (.scalp, "Afro"): set(s, 1.35, dir: n + V3(0, 0.35, 0), stiff: 0.55, curl: 1.2)
            case (.scalp, "Curly Long"): set(s, 3.6, dir: n * 0.4 + V3((r.x >= 0 ? 1 : -1) * 0.7, 0.1, -0.55), stiff: 0.18, curl: 0.8)
            case (.beard, "Clean Shaven"), (.mustache, "Clean Shaven"): set(s, 0)
            case (.beard, "Stubble"), (.mustache, "Stubble"): set(s, 0.05, stiff: 0.95)
            case (.beard, "Short Beard"): set(s, 0.22, stiff: 0.7)
            case (.mustache, "Short Beard"): set(s, 0.18, stiff: 0.7)
            case (.beard, "Full Beard"): set(s, (r.y > shape.lipTop ? 0.25 : 0.5) + max(0, shape.lipBottom - r.y) * 0.4, stiff: 0.4, curl: 0.35)
            case (.mustache, "Full Beard"): set(s, 0.42)
            case (.beard, "Goatee"):
                let ok = abs(r.x) < shape.mouthHalf + 0.05 && r.y < shape.lipBottom
                set(s, ok ? 0.65 : 0, stiff: 0.35)
            case (.mustache, "Goatee"): set(s, 0.35)
            case (.beard, "Handlebar"): set(s, 0)
            case (.mustache, "Handlebar"):
                let right = r.x > 0
                set(s, 0.35 + abs(r.x) * 1.4, dir: V3(right ? 1 : -1, -0.3, 0.4), stiff: 0.8, bend: right ? 2.6 : -2.6)
            case (.beard, "Wizard"): set(s, r.y > shape.lipTop ? 0.4 : 2.6 + max(0, shape.lipBottom - r.y) * 0.8, stiff: 0.15, curl: 0.15)
            case (.mustache, "Wizard"): set(s, 1.0, stiff: 0.2)
            default: break
            }
        }
        // let the new style settle under gravity before it is shown (no hair jutting out while it falls)
        Perf.time("presetSettle") { for _ in 0..<90 { simulate(f) } }
        for s in strands where s.length > 0 { for i in 0...s.n { s.prev[i] = s.pts[i] } }   // settled: start at rest
    }

    /// Explain why hair tools can't act yet.
    func notReady() {
        let msg: String
        if lastPixels == nil { msg = "⚠️ No camera image — allow camera access in System Settings › Privacy & Security › Camera" }
        else if face == nil { msg = "⚠️ No face found — face the camera with good light" }
        else { msg = "⏳ Still measuring your face — hold still for a moment" }
        toast = (msg, CACurrentMediaTime() + 3)
    }

    func growAll(_ kinds: Set<StrandKind>) {
        guard calibrated, let f = face else { notReady(); return }
        for s in strands where kinds.contains(s.kind) {
            if s.length == 0 { seed(s, f) }
            s.length = min(s.maxLen, s.length + s.maxLen * 0.25)
        }
    }

    /// Root position this frame: mustache follows the upper lip, chin beard follows the jaw.
    func rootLocal(_ s: Strand) -> V3 {
        if let q = s.skullQ { let k = shape.skull; return V3(k.c.x + q.x * k.r.x, k.c.y + q.y * k.r.y, k.c.z + q.z * k.r.z) }
        var r = s.root
        if let t = s.lipT {
            let lip = live.upperLipY(r.x) + 0.025
            r.y = lip + max(0.03, live.noseY - 0.07 - lip) * t
            if let z = shape.face.front(r.x, r.y)?.z { r.z = z }
        } else if let j = s.jaw, let q = live.beardPoint(j.t, j.inset) {
            r.x = q.x; r.y = q.y
            r.z = shape.face.front(q.x, q.y)?.z ?? shape.face.c.z
        }
        return r
    }

    /// World-space root. Beard roots come straight from the on-screen jawline (measured flat, so applying the
    /// head rotation again would shift them); everything else goes through the 3D head model.
    func rootWorld(_ s: Strand, _ f: LocalFrame) -> V3 {
        if let j = s.jaw, let q = live.beardPoint(j.t, j.inset) {
            let w = f.world(q)
            let z = shape.face.front(q.x, q.y)?.z ?? shape.face.c.z
            return V3(w.x, w.y, f.world3(V3(q.x, q.y, z)).z)
        }
        return f.world3(rootLocal(s))
    }

    private func anchor(_ s: Strand, _ f: LocalFrame) {
        let r = rootWorld(s, f)
        for i in 0...s.n { s.pts[i] = r; s.prev[i] = r }
    }
    private func seed(_ s: Strand, _ f: LocalFrame) { anchor(s, f); s.color = color }

    // MARK: geometry
    func imageRect() -> CGRect {
        let iw = imageSize.width, ih = imageSize.height
        let sc = max(bounds.width / iw, bounds.height / ih)
        let w = iw * sc, h = ih * sc
        return CGRect(x: (bounds.width - w) / 2, y: (bounds.height - h) / 2, width: w, height: h)
    }
    func map(_ p: CGPoint) -> CGPoint {   // normalized image → mirrored view
        let r = imageRect()
        return CGPoint(x: r.minX + (1 - p.x) * r.width, y: r.minY + p.y * r.height)
    }

    override func layout() {
        super.layout()
        CATransaction.begin(); CATransaction.setDisableActions(true)
        let r = imageRect()
        for l in [videoLayer, overlayLayer] as [CALayer] { l.bounds = CGRect(origin: .zero, size: r.size); l.position = CGPoint(x: r.midX, y: r.midY) }
        hairLayer.frame = bounds; cursorLayer.frame = bounds
        CATransaction.commit()
    }

    private func updateFace() {
        guard let lm = landmarks, CACurrentMediaTime() - lastFaceTime < 0.6 else {
            if let f = face { lostFrame = f }
            face = nil; return
        }
        let flat = LocalFrame(map(lm.eyeA), map(lm.eyeB))
        // head turn / nod from landmark cues relative to the neutral pose seen during calibration
        if let cue = poseCues(lm, flat, map) { debugCue = String(format: "%.2f %.2f", Double(cue.x), Double(cue.y)) }
        // head turn/nod from Vision's head-pose model (camera image is unmirrored, so yaw flips).
        // Filtered once per new camera frame with One Euro filters (smooth when still, responsive when moving).
        let newFrame = lastFaceTime != filteredAt
        let dt = filteredAt == 0 ? 1.0 / 30 : max(1.0 / 120, lastFaceTime - filteredAt)
        // turn-robust face size (eyes → mouth corners), once the face has been seen straight on
        let sizeV: CGFloat? = lm.mouthDrop(map).flatMap { d in camera?.dropRatio.map { d / $0 } }
        if newFrame, let vy = lm.vYaw, let vp = lm.vPitch {
            do {   // Vision's pose is absolute, so track it during calibration too (no snap when calibration ends)
                var ty = max(-1.2, min(1.2, -vy)), tp = max(-0.35, min(0.35, vp * pitchSign))
                // Vision under-reports strong turns (~0.9 when nearly in profile): the squeezed eye spacing
                // against the true face size tells the real angle
                if let sv = sizeV, sv > 1 {
                    let geo = acos(max(0.2, min(1, flat.scale / sv)))
                    ty = (ty < 0 ? -1 : 1) * min(1.35, abs(ty) + max(0, geo - abs(ty)) * dropWeight(ty))
                }
                // a single wild reading (glare, other face) is ignored; a real fast turn persists
                if abs(ty - yaw) > 0.5 && poseOutliers < 3 { poseOutliers += 1 } else {
                    poseOutliers = 0
                    yaw = CGFloat(fYaw(Double(ty), dt: dt))
                    pitch = CGFloat(fPitch(Double(tp), dt: dt))
                }
            }
        }
        if let amp = Env.vars["HAIRGAME_FAKEYAW"].flatMap(Double.init) {   // test hook
            yaw = CGFloat(amp) * sin(CGFloat(CACurrentMediaTime()) * 0.9); pitch = 0
        }
        // distance to the camera from apparent eye spacing (webcam ≈ 75° horizontal FOV)
        let eyePx = hypot((lm.eyeA.x - lm.eyeB.x) * imageSize.width, (lm.eyeA.y - lm.eyeB.y) * imageSize.height)
        let focal = imageSize.width / 2 / tan(75 * .pi / 360)
        let dist = max(2.5, min(40, focal * max(0.45, cos(yaw)) / max(1, eyePx)))
        camDistance = camDistance == 0 ? dist : camDistance + (dist - camDistance) * 0.2
        camera?.headDist = Float(camDistance)
        var target = flat
        target.camDist = camDistance
        // size from eye spacing, which shrinks by cos(turn); the turn angle comes from Vision's pose model
        target.scale = flat.scale / max(0.6, cos(yaw))
        if let sv = sizeV { target.scale += (max(target.scale, sv) - target.scale) * dropWeight(yaw) }   // strong turns
        debugIod = flat.scale; debugK = 1 / max(0.6, cos(yaw))
        camera?.headYaw = Float(-yaw); camera?.headPitch = Float(pitch)   // camera space is unmirrored
        let wasNil = face == nil
        // tracking stalled (e.g. Vision warming up) and the head moved meanwhile: carry the hair with the head
        // instead of letting the physics treat the catch-up as a violent head movement
        let stalled = newFrame && filteredAt > 0 && lastFaceTime - filteredAt > 0.15 && calibrated
        let beforeStall = face
        if wasNil { fOx.reset(); fOy.reset(); fScale.reset(); fRoll.reset() }
        if newFrame || wasNil {
            // filter in pixels; responsiveness scales with face size so it behaves the same near or far
            let u = Double(max(1, smoothed?.scale ?? target.scale))
            fOx.beta = 0.8 / u; fOy.beta = 0.8 / u
            smoothed = LocalFrame(origin: CGPoint(x: fOx(Double(target.origin.x), dt: dt), y: fOy(Double(target.origin.y), dt: dt)),
                                  scale: CGFloat(fScale(Double(target.scale), dt: dt)),
                                  roll: CGFloat(fRoll(Double(target.roll), dt: dt)), yaw: yaw, pitch: pitch)
            filteredAt = lastFaceTime
            if Env.vars["HAIRGAME_TRACE"] != nil, let sm = smoothed {
                FileHandle.standardError.write(String(format: "TRACE %.3f %.2f %.2f %.4f %.3f | %.2f %.2f %.4f %.3f\n", lastFaceTime,
                    Double(target.origin.x), Double(target.origin.y), Double(target.roll), Double(-(lm.vYaw ?? 0)),
                    Double(sm.origin.x), Double(sm.origin.y), Double(sm.roll), Double(yaw)).data(using: .utf8)!)
            }
        }
        face = smoothed ?? target
        face!.camDist = camDistance
        if stalled, let old = beforeStall, let f = face {
            for s in strands where s.length > 0 {
                for i in 0...s.n { s.pts[i] = f.world3(old.local3(s.pts[i])); s.prev[i] = s.pts[i] }
            }
        }
        let goodView = abs(lm.vYaw ?? 0) < 0.3 && abs(lm.vPitch ?? 0) < 0.3 && eyePx > 35
        if !calibrated && !goodView {
            calibrationHint = eyePx <= 35 ? "Move a little closer to the camera" : "Look straight at the camera so it can measure your face"
        } else if !calibrated {
            calibrationHint = nil
            var calFrame = target; calFrame.setRotation(roll: target.roll, yaw: 0, pitch: 0)
            let surf = shape.face
            let m = FaceShape(lm, edge: { calFrame.local(self.map($0)) }) { calFrame.unproject(self.map($0), onto: surf) }   // undo perspective
            if calibration == 0 { shape = m } else { shape.blend(m, 1 / CGFloat(calibration + 1)) }
            calibration += 1
            if calibrated {
                if let m = camera?.headMeasure {   // fit the 3D skull to the real head outline, so hair sits on the cap
                    let vol = CGFloat(camera?.hairVolume ?? 1)
                    shape.skullTop = m.top - CGFloat(Camera.thickness(1, Float(vol)))
                    shape.skullHalfW = m.halfW - CGFloat(Camera.thickness(0, Float(vol)))
                }
                // live measurements start from *this* frame in the live mapping (the calibration average uses a
                // slightly different mapping; switching between them made beard/mustache roots jump)
                let surfL = shape.face
                live = shape
                live.blend(FaceShape(lm, edge: { target.local(self.map($0)) }) { target.unproject(self.map($0), onto: surfL) }, 1)
                makeStrands(); strands.forEach { anchor($0, face!) }
            }
        } else {
            // lips/jaw measured back in head-model space (undoing turn, nod and perspective). The outline features
            // (jaw contour, brow line, beard centre) are on-screen measurements and stay current at any angle;
            // the front features are only trusted when not turned too far.
            let surf = shape.face
            let m = FaceShape(lm, edge: { target.local(self.map($0)) }) { target.unproject(self.map($0), onto: surf) }
            let k: CGFloat = wasNil ? 1 : 0.5                       // just re-found the face: take fresh measurements outright
            if abs(yaw) < 0.7 { live.blend(m, k) }
            else {
                if live.contour.count == m.contour.count { live.contour = zip(live.contour, m.contour).map { $0 + ($1 - $0) * k } } else { live.contour = m.contour }
                live.edgeBrowY = m.edgeBrowY ?? live.edgeBrowY; live.beardCenter = m.beardCenter ?? live.beardCenter
            }
            if debugTick % 60 == 0, Env.debug, let fr = face {
                let beard = strands.filter { $0.jaw != nil && $0.length > 0 }
                let l = beard.filter { rootWorld($0, fr).x < fr.origin.x }.count
                FileHandle.standardError.write("BEARD grown \(beard.count) left \(l) right \(beard.count - l)\n".data(using: .utf8)!)
                let flatC = lm.contour.map { flat.local(self.map($0)) }
                FileHandle.standardError.write(String(format: "JAW flat x %.2f..%.2f y %.2f | live x %.2f..%.2f y %.2f | calib x %.2f..%.2f | yaw %.2f pitch %.2f\n",
                    Double(flatC.map(\.x).min() ?? 0), Double(flatC.map(\.x).max() ?? 0), Double(flatC.map(\.y).min() ?? 0),
                    Double(m.contour.map(\.x).min() ?? 0), Double(m.contour.map(\.x).max() ?? 0), Double(m.contour.map(\.y).min() ?? 0),
                    Double(shape.contour.map(\.x).min() ?? 0), Double(shape.contour.map(\.x).max() ?? 0), Double(yaw), Double(pitch)).data(using: .utf8)!)
            }
        }
        if calibrated, let m = camera?.headMeasure {   // keep the 3D skull fitted to the real head outline (= the bald cap)
            let vol = CGFloat(camera?.hairVolume ?? 1)
            let top = m.top - CGFloat(Camera.thickness(1, Float(vol))), hw = m.halfW - CGFloat(Camera.thickness(0, Float(vol)))
            shape.skullTop = shape.skullTop == 0 ? top : shape.skullTop + (top - shape.skullTop) * 0.05
            shape.skullHalfW = shape.skullHalfW == 0 ? hw : shape.skullHalfW + (hw - shape.skullHalfW) * 0.05
        }
        if calibrated && wasNil, let f = face {
            if let old = lostFrame {
                // tracking dropped briefly (e.g. a big turn): carry the hair over in head space instead of resetting it
                for s in strands where s.length > 0 {
                    for i in 0...s.n { s.pts[i] = f.world3(old.local3(s.pts[i])); s.prev[i] = s.pts[i] }
                }
            } else { strands.forEach { anchor($0, f) } }
        }
    }

    // MARK: simulation
    /// Tell the bald cap where hair is grown (by direction around the skull) so it paints a matching hair base.
    private func publishCoverage() {
        guard let cam = camera, calibrated else { return }
        var tot = [Float](repeating: 0, count: 24), grown = [Float](repeating: 0, count: 24)
        var cr: CGFloat = 0, cg: CGFloat = 0, cb: CGFloat = 0, cn: CGFloat = 0
        let cy = shape.browY + 0.1
        for s in strands where s.kind == .scalp {
            let r = rootLocal(s)
            let th = atan2(r.y - cy, r.x)
            let b = max(0, min(23, Int((th + 0.3 * .pi) / (1.6 * .pi) * 24)))
            tot[b] += 1
            if s.length > 0.05 { grown[b] += 1; cr += s.color.r; cg += s.color.g; cb += s.color.b; cn += 1 }
        }
        var cov = (0..<24).map { tot[$0] > 0 ? grown[$0] / tot[$0] : 0 }
        // fill empty bins from neighbours and smooth a little
        for k in 0..<24 where tot[k] == 0 { cov[k] = 0.5 * (cov[max(0, k - 1)] + cov[min(23, k + 1)]) }
        cam.hairCoverage = (0..<24).map { k in (cov[max(0, k - 1)] + 2 * cov[k] + cov[min(23, k + 1)]) / 4 }
        if cn > 0 { cam.hairBase = (Float(cr / cn * 0.9), Float(cg / cn * 0.9), Float(cb / cn * 0.9)) }
    }

    private var lastTickT = 0.0
    private var lastRestart = 0.0
    /// Window minimised, fully covered or the app hidden: do no work at all (the camera stops processing too).
    private var offScreen: Bool {
        guard let w = window, !Env.harness else { return false }
        return w.isMiniaturized || !w.occlusionState.contains(.visible) || NSApp.isHidden
    }
    /// The camera delivered frames before but has gone quiet (sleep, another app took it, unplugged).
    private var cameraStalled: Bool {
        guard let cam = camera, lastPixels != nil, !Env.harness, !cam.paused else { return false }
        return CACurrentMediaTime() - cam.lastFrameAt > 2.5
    }
    private func tick() {
        let isHidden = offScreen
        if camera?.paused != isHidden { camera?.paused = isHidden }
        if isHidden { lastTickT = 0; return }
        if cameraStalled, CACurrentMediaTime() - lastRestart > 5 {
            lastRestart = CACurrentMediaTime(); camera?.restart(after: 0)
        }
        let tTick = CACurrentMediaTime()
        if lastTickT > 0 { Perf.add("tickGap", (tTick - lastTickT) * 1000) }
        lastTickT = tTick
        defer { Perf.add("tick", (CACurrentMediaTime() - tTick) * 1000); if Perf.on && debugTick % 120 == 0 { Perf.report() } }
        Perf.time("face") { updateFace() }
        if debugTick % 6 == 0 { publishCoverage() }
        if let f = face, calibrated {
            applyTool(f)
            Perf.time("sim") { simulate(f) }
        }
        stepParticles()
        render()
        debugTick += 1
        if glitchLog, let f = face, calibrated { glitchCheck(f) }
        let env = Env.vars
        if calibrated, let presets = env["HAIRGAME_PRESETS"], !harnessApplied {
            harnessApplied = true
            for p in presets.split(separator: "|") { applyPreset(String(p)) }
        }
        if env["HAIRGAME_NOBALD"] != nil { bald = false }
        if let dir = env["HAIRGAME_SNAPDIR"], calibrated, debugTick % Int(env["HAIRGAME_SNAPEVERY"] ?? "180")! == 0 { snapshot(to: URL(fileURLWithPath: dir)) }
        if debugTick % 60 == 0, Env.debug {
            let grown = strands.filter { $0.length > 0 }.count
            FileHandle.standardError.write(("dist=\(String(format: "%.1f", Double(camDistance))) iod=\(Int(debugIod)) k=\(String(format: "%.2f", Double(debugK))) scale=\(Int(face?.scale ?? 0)) vy=\(landmarks?.vYaw.map { String(format: "%.2f", Double($0)) } ?? "nil") vp=\(landmarks?.vPitch.map { String(format: "%.2f", Double($0)) } ?? "nil") cue=\(debugCue) pbias=\(String(format: "%.2f", Double(pitchBias))) yaw=\(String(format: "%.2f", Double(yaw))) pitch=\(String(format: "%.2f", Double(pitch))) cam=\(camera?.debugCam ?? -1) recv=\(debugRecv) face=\(face != nil) lm=\(landmarks != nil) calib=\(calibration) strands=\(strands.count) grown=\(grown) verts=\(builder.count) bounds=\(bounds.size) img=\(imageSize) mouse=\(mouse.map { "\(Int($0.x)),\(Int($0.y))" } ?? "nil") down=\(mouseIsDown) layer=\(hairLayer.frame) contents=\(hairLayer.contents != nil) origin=\(face.map { "\(Int($0.origin.x)),\(Int($0.origin.y)) s=\(Int($0.scale))" } ?? "-")\n").data(using: .utf8)!)
        }
    }
    private var debugTick = 0
    private let glitchLog = Env.vars["HAIRGAME_GLITCH"] != nil
    private var lastOrigin: CGPoint?, lastYawG: CGFloat = 0, lastGlitchSnap = 0
    /// Debug: report frames where hair moves violently, with what the tracking did at that moment.
    private func glitchCheck(_ f: LocalFrame) {
        let S = f.scale
        var maxV: CGFloat = 0, far = 0, nan = 0, worstKind = ""
        for st in strands where st.length > 0 {
            for i in 1...st.n {
                let p = st.pts[i]
                if p.x.isNaN || p.y.isNaN || p.z.isNaN { nan += 1; continue }
                let v = (p - st.prev[i]).len / S
                if v > maxV { maxV = v; worstKind = "\(st.kind)#\(i)" }
                if (p.xy - f.origin).len / S > 7 { far += 1 }
            }
        }
        let jump = lastOrigin.map { ($0 - f.origin).len / S } ?? 0
        let dyaw = abs(yaw - lastYawG)
        lastOrigin = f.origin; lastYawG = yaw
        if maxV > 0.25 || far > 0 || nan > 0 || jump > 0.3 || dyaw > 0.15 {
            FileHandle.standardError.write(String(format: "GLITCH tick %d maxV %.2f (%@) far %d nan %d originJump %.2f dYaw %.2f yaw %.2f pitch %.2f scale %.0f\n",
                debugTick, Double(maxV), worstKind, far, nan, Double(jump), Double(dyaw), Double(yaw), Double(pitch), Double(S)).data(using: .utf8)!)
            if debugTick - lastGlitchSnap > 20, let dir = Env.vars["HAIRGAME_SNAPDIR"] {
                lastGlitchSnap = debugTick; snapshot(to: URL(fileURLWithPath: dir))
            }
        }
    }
    private var harnessApplied = false
    private var spanLo: [CGFloat] = [], spanHi: [CGFloat] = []

    private func simulate(_ f: LocalFrame) {
        let S = f.scale
        let g = S * 0.0065
        let sk = shape.skull, fc = shape.face
        let roll = f.roll
        let t = CGFloat(CACurrentMediaTime())
        // the live on-screen face outline, sampled once per step (it was rebuilt for every hair point)
        let browL = live.edgeBrowY
        let spanY0 = live.chinY - 0.3, spanStep: CGFloat = 0.02
        let spanN = browL.map { max(0, Int(($0 - spanY0) / spanStep) + 2) } ?? 0
        if spanLo.count < spanN { spanLo = [CGFloat](repeating: .nan, count: spanN); spanHi = spanLo }
        for k in 0..<spanN {
            if let sp = live.faceSpan(atY: spanY0 + CGFloat(k) * spanStep) { spanLo[k] = sp.0; spanHi[k] = sp.1 } else { spanLo[k] = .nan }
        }
        for s in strands where s.length > 0.001 {
            let n = s.n
            let effLen = s.length * (1 - 0.35 * s.curl)
            let seg = effLen * S / CGFloat(n)
            let root = rootWorld(s, f)
            s.pts[0] = root; s.prev[0] = root
            // styled rest shape: growth direction with optional bend around the view axis
            var dir = f.worldDir(s.dir)
            var tp = root
            let reach = 0.2 + s.stiffness * s.stiffness * 1.6          // how far (in eye-widths) the style holds
            let bendStep = s.bend / CGFloat(n), cb = cos(bendStep), sb = sin(bendStep)
            for i in 1...n {
                tp = tp + dir * seg; targets[i] = tp
                dir = V3(dir.x * cb - dir.y * sb, dir.x * sb + dir.y * cb, dir.z)
            }
            let sway = sin(t * 1.3 + CGFloat(s.children[0].phase)) * S * 0.0003
            // coarse facial hair is heavily damped and slower than scalp hair (beards don't whip)
            let facial = s.kind != .scalp
            let vmax = (facial ? 0.09 : 0.18) * S              // no hair point moves more than this per frame
            let damping: CGFloat = facial ? 0.82 : 0.92
            for i in 1...n {
                var v = (s.pts[i] - s.prev[i]) * damping
                let vl = v.len
                if vl > vmax { v = v * (vmax / vl) }
                s.prev[i] = s.pts[i]
                var p = s.pts[i] + v + V3(sway, -g, 0)
                let dist = effLen * CGFloat(i) / CGFloat(n)
                let k = min(0.95, s.stiffness * exp(-pow(dist / reach, 2)) + s.curl * 0.04)
                p = p + (targets[i] - p) * k
                s.pts[i] = p
            }
            for _ in 0..<2 {
                for i in 1...n {
                    let before = s.pts[i]
                    var p = before
                    // collisions happen in head-model space, so they follow roll, turn and nod
                    // (one conversion in and out for both shapes; the round trip is exact)
                    let l0 = f.local3(p)
                    var lq = sk.pushOut(l0, pad: 1.04) ?? l0
                    if s.kind == .scalp, let q = fc.pushOut(lq, pad: 1.04) { lq = q }
                    if lq.x != l0.x || lq.y != l0.y || lq.z != l0.z { p = f.world3(lq) }
                    if s.kind == .scalp, let browL {
                        // keep hair off the *visible* face: below the brow line, inside the live on-screen face outline,
                        // slide it sideways to the outline edge on the side the strand grows from
                        var l = f.local(p.xy)
                        let k = Int(((l.y - spanY0) / spanStep).rounded())
                        if l.y < browL - 0.02, k >= 0, k < spanN, !spanLo[k].isNaN, l.x > spanLo[k] - 0.06, l.x < spanHi[k] + 0.06 {
                            // move toward the edge a little each frame (teleporting a whole face-width made hair flick out)
                            let target = s.root.x >= 0 ? spanHi[k] + 0.06 : spanLo[k] - 0.06
                            l.x += max(-0.08, min(0.08, target - l.x))
                            let w = f.world(l); p = V3(w.x, w.y, p.z)
                        }
                    }
                    if p.x != before.x || p.y != before.y || p.z != before.z {
                        // contact: the push must not become velocity (no launching), and friction lets hair rest
                        let vel = s.pts[i] - s.prev[i]
                        s.pts[i] = p
                        s.prev[i] = p - vel * 0.6
                    }
                }
                for i in 1...n {   // follow-the-leader length constraint
                    var d = s.pts[i] - s.pts[i - 1]
                    let L = d.len
                    d = L < 1e-4 ? (targets[i] - s.pts[i - 1]).norm : d * (1 / L)
                    let np = s.pts[i - 1] + d * seg
                    s.prev[i] = s.prev[i] + (np - s.pts[i]) * 0.9
                    s.pts[i] = np
                }
            }
        }
    }

    private func applyTool(_ f: LocalFrame) {
        guard let m = mouse else { return }
        let R = radius
        let drag = dragDelta; dragDelta = .zero
        func near(_ p: V3, _ r: CGFloat) -> Bool { (p.xy - m).len < r }

        if (tool == .brush || tool == .gel) && drag.len > 0.2 {
            let c = cos(-f.roll), s = sin(-f.roll)
            let ld = CGPoint(x: drag.x * c - drag.y * s, y: drag.x * s + drag.y * c).norm
            for st in strands where st.length > 0 {
                var best: CGFloat = 0
                for i in 1...st.n {
                    let d = (st.pts[i].xy - m).len
                    if d < R {
                        let fall = 1 - d / R
                        st.pts[i] = st.pts[i] + V3(drag.x, drag.y, 0) * (0.85 * fall)
                        best = max(best, fall)
                    }
                }
                if best > 0 {
                    let want = V3(ld.x, ld.y, min(0, st.dir.z) * 0.5).norm
                    st.dir = (st.dir + (want - st.dir) * (0.16 * best)).norm
                    if tool == .brush { st.curl *= 0.97; st.bend *= 0.95 }
                }
            }
        }
        guard mouseIsDown else { return }

        switch tool {
        case .brush: break
        case .serum:
            for s in strands {
                var hit = near(rootWorld(s, f), R * 1.2)
                if !hit && s.length > 0 { hit = s.pts.contains { near($0, R) } }
                guard hit else { continue }
                if s.length == 0 { seed(s, f) }
                s.length = min(s.maxLen * s.lenVar, s.length + 0.022 * s.lenVar)
            }
            for _ in 0..<5 {
                particles.append(Particle(p: m + CGPoint(x: .random(in: -R...R) * 0.6, y: .random(in: -R...R) * 0.6),
                                          v: CGPoint(x: .random(in: -1.5...1.5), y: .random(in: -1...2.5)),
                                          color: RGB(0.35, 1.0, 0.6), life: 0.9, size: .random(in: 2...5)))
            }
        case .razor:
            for s in strands where s.length > 0 {
                if near(rootWorld(s, f), R) { clip(s, from: 0) }
                else if let i = (1...s.n).first(where: { near(s.pts[$0], R * 0.6) }) { clip(s, from: i) }
            }
        case .scissors:
            for s in strands where s.length > 0 {
                if let i = (1...s.n).first(where: { near(s.pts[$0], R * 0.5) }) { clip(s, from: i) }
            }
        case .dye:
            for s in strands where s.length > 0 && s.pts.contains(where: { near($0, R) }) { s.color = s.color.mix(color, 0.15) }
            if Bool.random() {
                particles.append(Particle(p: m, v: CGPoint(x: .random(in: -2...2), y: .random(in: -2...1)), color: color, life: 0.6, size: .random(in: 3...6)))
            }
        case .gel:
            for s in strands where s.length > 0 && s.pts.contains(where: { near($0, R) }) { s.stiffness = min(0.95, s.stiffness + 0.025) }
        case .curl:
            for s in strands where s.length > 0 && s.pts.contains(where: { near($0, R) }) { s.curl = min(1.3, s.curl + 0.02) }
        case .blow:
            let BR = R * 3
            for s in strands where s.length > 0 {
                for i in 1...s.n {
                    let d = s.pts[i].xy - m, L = d.len
                    if L < BR && L > 1 { let k = (1 - L / BR) * 4 / L; s.pts[i] = s.pts[i] + V3(d.x * k, d.y * k, 0) }
                }
            }
            particles.append(Particle(p: m, v: CGPoint(x: .random(in: -5...5), y: .random(in: -5...5)), color: RGB(0.9, 0.95, 1), life: 0.4, size: 2))
        }
    }

    /// Cut a strand at segment i; the removed pieces fall away as physics clippings.
    private func clip(_ s: Strand, from i: Int) {
        let start = max(i, 0)
        if start < s.n && particles.count < 2500 {
            for j in start..<s.n {
                let a = s.pts[j].xy, b = s.pts[j + 1].xy, d = b - a
                for _ in 0..<2 {
                    particles.append(Particle(p: (a + b) * 0.5 + CGPoint(x: .random(in: -3...3), y: .random(in: -3...3)),
                                              v: (s.pts[j] - s.prev[j]).xy + CGPoint(x: .random(in: -1.5...1.5), y: .random(in: 0...2)),
                                              color: s.color, life: 2.2, size: max(2, d.len),
                                              angle: atan2(d.y, d.x), spin: .random(in: -0.2...0.2), isHair: true))
                }
            }
        }
        s.length = start == 0 ? 0 : s.length * CGFloat(start - 1) / CGFloat(s.n)
        if s.length < 0.02 { s.length = 0 }
    }

    private func stepParticles() {
        let g = (face?.scale ?? 100) * 0.005
        for i in particles.indices {
            particles[i].v = particles[i].v * 0.985 + CGPoint(x: 0, y: -g * (particles[i].isHair ? 1 : 0.4))
            particles[i].p = particles[i].p + particles[i].v
            particles[i].angle += particles[i].spin
            particles[i].life -= 1.0 / 60
        }
        particles.removeAll { $0.life <= 0 || $0.p.y < -40 }
    }

    // MARK: rendering
    private struct GroupKey: Hashable { var color: RGB; var w: Int }

    private func render() {
        CATransaction.begin(); CATransaction.setDisableActions(true)
        updateCursor()
        updateHUD()
        CATransaction.commit()
        guard bounds.width > 1, let gpu else { return }
        let scale = window?.backingScaleFactor ?? 2
        builder.reset(scale)
        if let f = face, calibrated { Perf.time("geom") { buildHair(f) } }
        buildFX()
        Perf.time("gpu") { gpu.draw(builder.ptr, builder.count, viewSize: bounds.size, scale: scale) }
        Perf.add("verts(k)", Double(builder.count) / 1000)
    }

    private func buildHair(_ f: LocalFrame) {
        let S = f.scale, px = S / 95
        let sk = shape.skull, fc = shape.face
        let Lx: Float = -0.42, Ly: Float = 0.91          // light from upper left (view space)
        let view = f.viewDirLocal
        let skOcc = Ellipsoid(c: sk.c, r: sk.r * 0.94), fcOcc = Ellipsoid(c: fc.c, r: fc.r * 0.94)   // hair rests at 1.04
        // lower head and neck (jaw to nape): hides hair hanging down the far side when turned
        let hw = shape.halfWidth, chin = shape.chinY
        let nkOcc = Ellipsoid(c: V3(0, chin - 0.1, -0.75), r: V3(hw * 0.82, 1.05, 0.95))
        // back-to-front by depth toward the camera (changes as the head turns)
        if depthKey.count != drawOrder.count { depthKey = [CGFloat](repeating: 0, count: drawOrder.count) }
        var order = Array(drawOrder.indices)
        for i in drawOrder.indices { depthKey[i] = f.rotate(drawOrder[i].root).z }
        order.sort { depthKey[$0] < depthKey[$1] }
        drawOrder = order.map { drawOrder[$0] }
        // dense base layer at the roots, drawn under all strands: real hair hides the scalp completely
        var basePts = [CGPoint](repeating: .zero, count: 3)
        for s in drawOrder where s.kind == .scalp && s.length > 0.04 {
            let l0 = f.local3(s.pts[0])
            guard !(skOcc.occludes(l0, view) || fcOcc.occludes(l0, view) || nkOcc.occludes(l0, view)) else { continue }
            let reach = min(s.n, 3)
            basePts[0] = s.pts[0].xy; basePts[1] = s.pts[max(1, reach / 2)].xy; basePts[2] = s.pts[reach].xy
            let c = s.color, w = min(0.2, 0.08 + s.length * 0.12) * S
            basePts.withUnsafeBufferPointer { b in
                // tapered and fading toward its end: blunt thick ends lined up into a blocky edge along the hairline
                builder.add(b, width: w, taper: 0.75) { _, t in (Float(c.r * 0.55), Float(c.g * 0.55), Float(c.b * 0.55), 0.92 * (1 - 0.75 * Float(t) * Float(t))) }
            }
        }
        for s in drawOrder where s.length > 0.001 {
            let n = s.n
            if s.kind == .scalp {
                for i in 0...n {   // hidden when the head is between this point and the camera
                    let l = f.local3(s.pts[i])
                    vis[i] = !(skOcc.occludes(l, view) || fcOcc.occludes(l, view) || nkOcc.occludes(l, view))
                }
            } else {
                // beard & mustache roots follow the *visible* landmarks, so they are never behind the face;
                // the coarse face ellipsoid would wrongly hide the jaw sides when the head tips down
                for i in 0...n { vis[i] = true }
            }
            // Catmull-Rom subdivision ×2 of the guide
            var m = 0
            for i in 0..<n {
                let p0 = s.pts[max(0, i - 1)].xy, p1 = s.pts[i].xy, p2 = s.pts[i + 1].xy, p3 = s.pts[min(n, i + 2)].xy
                pts2[m] = p1; m += 1
                pts2[m] = (p1 + p2) * 0.5625 - (p0 + p3) * 0.0625; m += 1
            }
            pts2[m] = s.pts[n].xy; m += 1

            // visible runs of the subdivided polyline
            runs.removeAll(keepingCapacity: true)
            var start = -1
            for j in 0..<m {
                if vis[j / 2] { if start < 0 { start = j } } else if start >= 0 { runs.append((start, j)); start = -1 }
            }
            if start >= 0 { runs.append((start, m)) }
            if runs.isEmpty { continue }

            // soft contact shadow under the clump
            pts2.withUnsafeBufferPointer { buf in
                for (a, b) in runs where b - a >= 2 {
                    builder.add(UnsafeBufferPointer(rebasing: buf[a..<b]), width: 3.4 * px, taper: 0.4) { _, t in (0, 0, 0, 0.16 * (1 - t * 0.6)) }
                }
            }

            // per-point data shared by every child hair of this clump
            let inv = 1 / CGFloat(m - 1)
            for j in 0..<m {
                let pa = pts2[max(0, j - 1)], pb = pts2[min(m - 1, j + 1)]
                var tx = Float(pb.x - pa.x), ty = Float(pb.y - pa.y)
                let l = max(1e-4, (tx * tx + ty * ty).squareRoot()); tx /= l; ty /= l
                nrm[j] = CGPoint(x: CGFloat(ty), y: CGFloat(-tx))
                let d = tx * Lx + ty * Ly                               // anisotropic (Kajiya-style) highlight
                let x = max(0, 1 - d * d), x2 = x * x, x4 = x2 * x2
                spec[j] = x4 * x4 * x4
                let t = Float(j) * Float(inv)
                lum[j] = 0.62 + 0.5 * t                                 // darker roots, lighter ends
                alph[j] = 0.95 - 0.25 * t * t
                let tc = CGFloat(t)
                taperT[j] = (1 - 0.85 * tc) * min(1, tc * 6 + 0.35)
            }
            // where a strand passes behind the head it fades out over a few points instead of ending in a blunt
            // cut (blunt cuts lined up into a jagged staircase along the hairline)
            if runs.count > 1 || runs[0].0 > 0 || runs[0].1 < m {
                var last = -100
                for j in 0..<m { if !vis[j / 2] { last = j }; fadeTmp[j] = Float(j - last) }
                last = 100_000
                for j in stride(from: m - 1, through: 0, by: -1) { if !vis[j / 2] { last = j }; fadeTmp[j] = min(fadeTmp[j], Float(last - j)) }
                for j in 0..<m { alph[j] *= min(1, fadeTmp[j] / 4) }
            }
            let base = s.color
            let curlAmp = s.curl * 0.09 * S
            let freq = 2 * CGFloat.pi / 0.24 * s.length * inv
            for ch in s.children {
                let off = ch.offset * S
                for j in 0..<m {
                    var o = off * taperT[j]
                    if curlAmp != 0 { o += curlAmp * sin(CGFloat(j) * freq + ch.phase) * min(1, CGFloat(j) * inv * 4) }
                    cp[j] = CGPoint(x: pts2[j].x - nrm[j].x * o, y: pts2[j].y - nrm[j].y * o)
                }
                let rgb = (Float(base.r * ch.shade), Float(base.g * ch.shade), Float(base.b * ch.shade))
                let shine = Float(0.25 + 0.35 * (ch.shade - 0.72) / 0.48)
                for (a, b) in runs {
                    builder.ribbon(a, b, m: m, pts: cp, nrm: nrm, lum: lum, spec: spec, alpha: alph,
                                   rgb: rgb, shine: shine, width: s.width * px, taper: 0.6)
                }
            }
        }
    }

    private func buildFX() {
        var seg = [CGPoint](repeating: .zero, count: 2)
        for p in particles {
            let a = Float(min(1, p.life * 2))
            let c = p.color
            if p.isHair {
                let d = CGPoint(x: cos(p.angle), y: sin(p.angle)) * (p.size / 2)
                seg[0] = p.p - d; seg[1] = p.p + d
                seg.withUnsafeBufferPointer { builder.add($0, width: 1.0, taper: 0) { _, _ in (Float(c.r), Float(c.g), Float(c.b), a) } }
            } else {
                seg[0] = p.p - CGPoint(x: p.size * 0.15, y: 0); seg[1] = p.p + CGPoint(x: p.size * 0.15, y: 0)
                seg.withUnsafeBufferPointer { builder.add($0, width: p.size, taper: 0) { _, _ in (Float(c.r), Float(c.g), Float(c.b), a * 0.7) } }
            }
        }
    }

    private func updateCursor() {
        if let m = mouse {
            let R = tool == .blow ? radius * 3 : radius
            cursorLayer.path = CGPath(ellipseIn: CGRect(x: m.x - R, y: m.y - R, width: 2 * R, height: 2 * R), transform: nil)
            cursorLayer.lineDashPattern = tool == .blow ? [4, 4] : nil
            iconLayer.string = tool.icon; iconLayer.position = CGPoint(x: m.x + 8, y: m.y + 6); iconLayer.isHidden = false
        } else { cursorLayer.path = nil; iconLayer.isHidden = true }
    }

    private func updateHUD() {
        var text: String
        if lastPixels == nil {
            text = status != "Waiting for camera…" ? status
                : CACurrentMediaTime() - startedAt > 6 ? "📷 No video yet — check camera access in System Settings, or close other apps using the camera"
                : "Waiting for camera… (allow camera access if macOS asks)"
        }
        else if cameraStalled { text = "📷 Camera stopped — reconnecting… (close other apps using the camera)" }
        else if face == nil { text = "Looking for your face… 👀" }
        else if !calibrated {
            text = calibrationHint ?? "Measuring your face — hold still… \(Int(Double(calibration) / Double(calibrationFrames) * 100))%"
        }
        else { text = "\(tool.icon) \(tool.name): \(tool.hint)" }
        if let (msg, until) = toast, CACurrentMediaTime() < until { text = msg }
        let s = " \(text) "
        if (hudLayer.string as? String) != s {
            hudLayer.string = s
            let w = (s as NSString).size(withAttributes: [.font: NSFont.systemFont(ofSize: 15, weight: .semibold)]).width + 4
            hudLayer.bounds = CGRect(x: 0, y: 0, width: w, height: 22)
        }
        hudLayer.position = CGPoint(x: 14, y: bounds.height - 34)
        keysLayer.bounds = CGRect(x: 0, y: 0, width: 330, height: 17)
        keysLayer.position = CGPoint(x: 14, y: bounds.height - 56)
    }

    // MARK: snapshot (rendered at camera resolution, encoded off the main thread)
    func snapshot(to folder: URL? = nil) {
        func fail(_ m: String) { toast = ("⚠️ Snapshot failed: \(m)", CACurrentMediaTime() + 3); FileHandle.standardError.write("snapshot failed: \(m)\n".data(using: .utf8)!) }
        guard let pb = lastPixels else { return fail("no camera frame") }
        guard let cam = camera, let still = cam.cgImage(pb) else { return fail("could not read camera frame") }
        let W = still.width, H = still.height
        guard let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue) else { return fail("no bitmap") }
        let full = CGRect(x: 0, y: 0, width: W, height: H)
        ctx.saveGState(); ctx.translateBy(x: CGFloat(W), y: 0); ctx.scaleBy(x: -1, y: 1)
        ctx.draw(still, in: full)
        if bald, face != nil, let ov = overlayImage { ctx.interpolationQuality = .high; ctx.draw(ov, in: full) }
        if Env.vars["HAIRGAME_LMDRAW"] != nil, let lm = landmarks {   // debug: landmark dots
            func dots(_ p: [CGPoint], _ c: CGColor) { ctx.setFillColor(c); for q in p { ctx.fillEllipse(in: CGRect(x: q.x * CGFloat(W) - 3, y: q.y * CGFloat(H) - 3, width: 6, height: 6)) } }
            dots(lm.contour, CGColor(red: 0, green: 1, blue: 0, alpha: 1)); dots(lm.brows, CGColor(red: 0, green: 0, blue: 1, alpha: 1))
            dots([lm.eyeA, lm.eyeB], CGColor(red: 1, green: 0, blue: 0, alpha: 1)); dots(lm.lips, CGColor(red: 1, green: 1, blue: 0, alpha: 1))
        }
        ctx.restoreGState()
        // hair lives in view coordinates: render it onto the image rect at camera resolution
        let r = imageRect()
        builder.reset(CGFloat(W) / r.width)
        if let f = face, calibrated { buildHair(f) }
        buildFX()
        if let hairImg = gpu?.image(builder.ptr, builder.count, rect: r, W: W, H: H) { ctx.draw(hairImg, in: full) }
        guard let img = ctx.makeImage() else { return fail("no image") }
        if folder == nil { toast = ("📸 Saved to Documents › hair game › Snapshots", CACurrentMediaTime() + 2.5) }
        let tickN = debugTick
        DispatchQueue.global(qos: .utility).async {
            let dir = folder ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents/hair game/Snapshots")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
            var url = dir.appendingPathComponent("Hair \(f.string(from: Date())).jpg")
            if folder != nil { url = dir.appendingPathComponent(String(format: "t%05d.jpg", tickN)) }   // test runs: one file per tick
            if let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) {
                CGImageDestinationAddImage(dest, img, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
                if !CGImageDestinationFinalize(dest) { FileHandle.standardError.write("snapshot write failed \(url.path)\n".data(using: .utf8)!) }
            } else { FileHandle.standardError.write("snapshot dest failed \(url.path)\n".data(using: .utf8)!) }
        }
    }

    // MARK: input
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseMoved, .mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self))
    }
    override func mouseMoved(with e: NSEvent) { mouse = convert(e.locationInWindow, from: nil) }
    override func mouseExited(with e: NSEvent) { if !mouseIsDown { mouse = nil } }
    override func mouseDown(with e: NSEvent) { if !calibrated || face == nil { notReady() }; window?.makeFirstResponder(self); mouseIsDown = true; mouse = convert(e.locationInWindow, from: nil) }
    override func mouseDragged(with e: NSEvent) {
        let p = convert(e.locationInWindow, from: nil)
        if let m = mouse { dragDelta = dragDelta + (p - m) }
        mouse = p
    }
    override func mouseUp(with e: NSEvent) {
        mouseIsDown = false
        let p = convert(e.locationInWindow, from: nil)
        if !bounds.contains(p) { mouse = nil }                    // released outside: no stray cursor or tool
    }
    override func scrollWheel(with e: NSEvent) {
        radius = min(160, max(8, radius - e.scrollingDeltaY * 0.5))
        onRadiusChanged?(radius)
    }
    override func keyDown(with e: NSEvent) {
        guard let ch = e.charactersIgnoringModifiers?.lowercased() else { return }
        if let n = Int(ch), let t = Tool(rawValue: n - 1) { tool = t; onToolChanged?(t) }
        else if ch == "b" { bald.toggle(); onBaldChanged?(bald) }
        else if ch == " " { snapshot() }
        else { super.keyDown(with: e) }
    }
}

// MARK: - Color swatch

final class Swatch: NSView {
    let color: NSColor; let action: (NSColor) -> Void
    var selected = false { didSet { if selected != oldValue { needsDisplay = true } } }
    init(_ c: NSColor, _ a: @escaping (NSColor) -> Void) {
        color = c; action = a
        super.init(frame: NSRect(x: 0, y: 0, width: 22, height: 22))
        translatesAutoresizingMaskIntoConstraints = false
        widthAnchor.constraint(equalToConstant: 22).isActive = true
        heightAnchor.constraint(equalToConstant: 22).isActive = true
        toolTip = "Use this color for dye and new growth"
    }
    required init?(coder: NSCoder) { fatalError() }
    override func draw(_ r: NSRect) {
        let p = NSBezierPath(ovalIn: bounds.insetBy(dx: 3, dy: 3))
        color.setFill(); p.fill()
        NSColor.white.withAlphaComponent(0.35).setStroke(); p.lineWidth = 1; p.stroke()
        if selected {
            let ring = NSBezierPath(ovalIn: bounds.insetBy(dx: 0.75, dy: 0.75))
            NSColor.controlAccentColor.setStroke(); ring.lineWidth = 1.5; ring.stroke()
        }
    }
    override func resetCursorRects() { addCursorRect(bounds, cursor: .pointingHand) }
    override func mouseDown(with e: NSEvent) { action(color) }
}

/// The bar under the video: solid dark background with a hairline on top.
final class Panel: NSView {
    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0.115, alpha: 1).cgColor
        let line = CALayer(); line.backgroundColor = NSColor(white: 1, alpha: 0.09).cgColor
        line.autoresizingMask = [.layerWidthSizable, .layerMinYMargin]
        line.frame = CGRect(x: 0, y: frame.height - 1, width: frame.width, height: 1)
        layer?.addSublayer(line); hairline = line
    }
    private var hairline: CALayer?
    override func layout() { super.layout(); hairline?.frame = CGRect(x: 0, y: bounds.height - 1, width: bounds.width, height: 1) }
    required init?(coder: NSCoder) { fatalError() }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    let hair = HairView(frame: .zero)
    let camera = Camera()
    let tools = NSSegmentedControl(labels: Tool.allCases.map { "\($0.icon) \($0.name)" }, trackingMode: .selectOne, target: nil, action: nil)
    let baldSwitch = NSSwitch()
    let well = NSColorWell(style: .minimal)
    let hairPopup = NSPopUpButton(), facePopup = NSPopUpButton()
    private var sizeSlider: NSSlider!
    private var swatches: [Swatch] = []
    private let capPopover = NSPopover()

    static let palette: [NSColor] = [
        .init(srgbRed: 0.06, green: 0.05, blue: 0.05, alpha: 1), .init(srgbRed: 0.24, green: 0.14, blue: 0.07, alpha: 1),
        .init(srgbRed: 0.52, green: 0.34, blue: 0.19, alpha: 1), .init(srgbRed: 0.93, green: 0.8, blue: 0.5, alpha: 1),
        .init(srgbRed: 0.72, green: 0.27, blue: 0.1, alpha: 1), .init(srgbRed: 0.6, green: 0.6, blue: 0.6, alpha: 1),
        .init(srgbRed: 0.96, green: 0.96, blue: 0.94, alpha: 1), .init(srgbRed: 1.0, green: 0.4, blue: 0.7, alpha: 1),
        .init(srgbRed: 0.25, green: 0.5, blue: 1.0, alpha: 1), .init(srgbRed: 0.2, green: 0.85, blue: 0.4, alpha: 1),
        .init(srgbRed: 0.6, green: 0.3, blue: 0.95, alpha: 1)]

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1300, height: 880),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Hair Game"
        window.appearance = NSAppearance(named: .darkAqua)
        window.minSize = NSSize(width: 1120, height: 660)
        window.setFrameAutosaveName("HairGameMain")              // reopens where you left it
        window.delegate = self
        window.collectionBehavior.insert(.fullScreenPrimary)

        hair.camera = camera
        hair.onToolChanged = { [weak self] t in self?.tools.selectedSegment = t.rawValue }
        hair.onBaldChanged = { [weak self] b in self?.baldSwitch.state = b ? .on : .off }
        hair.onRadiusChanged = { [weak self] r in self?.sizeSlider.doubleValue = Double(r) }

        let panel = buildPanel()
        let content = NSView()
        for v in [hair, panel] as [NSView] { v.translatesAutoresizingMaskIntoConstraints = false; content.addSubview(v) }
        NSLayoutConstraint.activate([
            hair.topAnchor.constraint(equalTo: content.topAnchor),
            hair.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            hair.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            hair.bottomAnchor.constraint(equalTo: panel.topAnchor),
            panel.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            panel.bottomAnchor.constraint(equalTo: content.bottomAnchor),
        ])
        window.contentView = content
        if !window.setFrameUsingName("HairGameMain") { window.center() }
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(hair)
        NSApp.activate(ignoringOtherApps: true)

        camera.onFrame = { [weak self] fr in self?.hair.receive(fr) }
        camera.onError = { [weak self] msg in self?.hair.status = msg; self?.hair.toast = ("⚠️ " + msg, CACurrentMediaTime() + 4) }
        camera.onRecordDone = { [weak self] url in
            self?.hair.toast = (url != nil ? "🎬 Test clip saved to Documents › hair game › Clips" : "⚠️ Recording failed", CACurrentMediaTime() + 3)
        }
        let env = Env.vars
        if let dir = env["HAIRGAME_SELFTEST"] { camera.selfTest(URL(fileURLWithPath: dir)); exit(0) }
        if env["HAIRGAME_SYNTH"] != nil {
            camera.startSynthetic()
            if let t = env["HAIRGAME_RECORDTEST"] { camera.startRecording(to: URL(fileURLWithPath: t), seconds: Double(env["HAIRGAME_RECSECS"] ?? "3") ?? 3) }
        }
        else if let clip = env["HAIRGAME_REPLAY"] {
            camera.startReplay(URL(fileURLWithPath: clip))
            if let t = env["HAIRGAME_RECORDTEST"] { camera.startRecording(to: URL(fileURLWithPath: t), seconds: Double(env["HAIRGAME_RECSECS"] ?? "3") ?? 3) }
        } else { camera.start() }
        if let path = env["HAIRGAME_UISHOT"] {   // debug: image of the window's controls (video/Metal layers excluded)
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in
                guard let v = self?.window.contentView, let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { return }
                v.cacheDisplay(in: v.bounds, to: rep)
                try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                if env["HAIRGAME_UISHOT_POPOVER"] != nil, let pv = self?.capPopover.contentViewController?.view {
                    pv.appearance = NSAppearance(named: .darkAqua)
                    pv.setFrameSize(pv.fittingSize); pv.layoutSubtreeIfNeeded()
                    guard let r2 = pv.bitmapImageRepForCachingDisplay(in: pv.bounds) else { return }
                    pv.cacheDisplay(in: pv.bounds, to: r2)
                    try? r2.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path + ".popover.png"))
                }
            }
        }
    }

    /// Two tidy rows: tools · brush size · colours, then styles · bald cap · snapshot.
    private func buildPanel() -> NSView {
        tools.target = self; tools.action = #selector(toolPicked); tools.selectedSegment = hair.tool.rawValue
        tools.segmentDistribution = .fillEqually
        for t in Tool.allCases { tools.setToolTip(t.hint, forSegment: t.rawValue) }

        sizeSlider = NSSlider(value: Double(hair.radius), minValue: 8, maxValue: 160, target: self, action: #selector(sizeChanged(_:)))
        sizeSlider.controlSize = .small
        sizeSlider.widthAnchor.constraint(equalToConstant: 96).isActive = true
        sizeSlider.toolTip = "Brush size (or scroll over the picture)"

        swatches = Self.palette.map { c in Swatch(c) { [weak self] c in self?.setColor(c) } }
        swatches[1].selected = true
        well.color = Self.palette[1]; well.target = self; well.action = #selector(wellChanged)
        well.toolTip = "Pick any color"
        let swatchRow = NSStackView(views: swatches + [well]); swatchRow.spacing = 0

        let row1 = NSStackView()
        row1.setViews([tools], in: .leading)
        row1.setViews([caption("Size"), sizeSlider, separator(), caption("Color"), swatchRow], in: .trailing)

        baldSwitch.state = .on; baldSwitch.target = self; baldSwitch.action = #selector(baldToggled)
        baldSwitch.controlSize = .small
        baldSwitch.toolTip = "Hide your real hair under a bald cap (B)"
        let capButton = symbolButton("slider.horizontal.3", "Bald cap settings", #selector(showCapSettings(_:)))
        let snap = button("Snapshot", "camera.fill", #selector(snapshot))
        snap.bezelColor = .controlAccentColor; snap.keyEquivalent = ""
        snap.toolTip = "Save a picture to Documents › hair game › Snapshots (space)"

        let row2 = NSStackView()
        row2.setViews([popup(hairPopup, "Hairstyle", "scissors", ["Bald", "Buzz Cut", "Crew Cut", "Bob", "Side Part", "Slicked Back", "Long", "Curly Long", "Afro", "Spiky", "Mohawk"]),
                       popup(facePopup, "Facial hair", "mustache", ["Clean Shaven", "Stubble", "Short Beard", "Full Beard", "Goatee", "Handlebar", "Wizard"]),
                       separator(),
                       button("Grow Hair", "sparkles", #selector(growHair)), button("Grow Beard", "sparkles", #selector(growBeard)),
                       button("Shave All", "eraser", #selector(shaveAll)), button("Reset Style", "arrow.uturn.backward", #selector(resetStyle))],
                      in: .leading)
        row2.setViews([caption("Bald cap"), baldSwitch, capButton, separator(), snap], in: .trailing)

        for r in [row1, row2] { r.orientation = .horizontal; r.spacing = 10; r.alignment = .centerY }
        let stack = NSStackView(views: [row1, row2])
        stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 10
        stack.edgeInsets = NSEdgeInsets(top: 12, left: 14, bottom: 12, right: 14)
        for r in [row1, row2] { r.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -28).isActive = true }

        let panel = Panel(frame: NSRect(x: 0, y: 0, width: 1300, height: 90))
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: panel.topAnchor), stack.bottomAnchor.constraint(equalTo: panel.bottomAnchor),
            stack.leadingAnchor.constraint(equalTo: panel.leadingAnchor), stack.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
        ])
        buildCapPopover()
        return panel
    }

    /// Advanced bald-cap tuning, out of the way in a popover.
    private func buildCapPopover() {
        let head = NSSlider(value: Double(camera.hairVolume), minValue: 0.3, maxValue: 2.0, target: self, action: #selector(headChanged(_:)))
        let strength = NSSlider(value: Double(camera.strength), minValue: 0.4, maxValue: 2.0, target: self, action: #selector(strengthChanged(_:)))
        for s in [head, strength] { s.widthAnchor.constraint(equalToConstant: 200).isActive = true }
        func row(_ title: String, _ s: NSSlider, _ help: String) -> NSView {
            let t = NSTextField(labelWithString: title); t.font = .systemFont(ofSize: 12, weight: .semibold)
            let h = NSTextField(wrappingLabelWithString: help); h.font = .systemFont(ofSize: 11); h.textColor = .secondaryLabelColor
            h.preferredMaxLayoutWidth = 230
            let v = NSStackView(views: [t, s, h]); v.orientation = .vertical; v.alignment = .leading; v.spacing = 4
            return v
        }
        let title = NSTextField(labelWithString: "Bald cap"); title.font = .systemFont(ofSize: 13, weight: .bold)
        let recal = button("Recalibrate Face", "face.smiling", #selector(recalibrate))
        let rec = button("Record Test Clip", "record.circle", #selector(recordClip))
        let v = NSStackView(views: [title,
                                    row("Hair volume", head, "How thick your real hair is. More makes the bald scalp smaller inside your hair."),
                                    row("Hair removal", strength, "How aggressively real hair is removed."),
                                    recal, rec])
        v.orientation = .vertical; v.alignment = .leading; v.spacing = 12
        v.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        let vc = NSViewController(); vc.view = v
        capPopover.contentViewController = vc
        capPopover.behavior = .transient
        capPopover.appearance = NSAppearance(named: .darkAqua)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { camera.stopRecordingForQuit() }

    private func caption(_ s: String) -> NSTextField {
        let t = NSTextField(labelWithString: s); t.font = .systemFont(ofSize: 11, weight: .medium); t.textColor = .secondaryLabelColor
        return t
    }
    private func separator() -> NSView {
        let b = NSBox(); b.boxType = .separator
        b.translatesAutoresizingMaskIntoConstraints = false
        b.widthAnchor.constraint(equalToConstant: 1).isActive = true
        b.heightAnchor.constraint(equalToConstant: 18).isActive = true
        return b
    }
    private func button(_ t: String, _ symbol: String, _ a: Selector) -> NSButton {
        let b = NSButton(title: t, target: self, action: a)
        if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: t) { b.image = img; b.imagePosition = .imageLeading }
        return b
    }
    private func symbolButton(_ symbol: String, _ tip: String, _ a: Selector) -> NSButton {
        let b = NSButton(title: "", target: self, action: a)
        b.image = NSImage(systemSymbolName: symbol, accessibilityDescription: tip) ?? NSImage(named: NSImage.actionTemplateName)
        b.imagePosition = .imageOnly; b.toolTip = tip
        return b
    }
    private func popup(_ p: NSPopUpButton, _ title: String, _ symbol: String, _ items: [String]) -> NSPopUpButton {
        p.pullsDown = true
        p.addItem(withTitle: title)
        if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: title) { p.item(at: 0)?.image = img }
        p.addItems(withTitles: items)
        p.target = self; p.action = #selector(presetPicked(_:))
        return p
    }
    private func setColor(_ c: NSColor) {
        hair.color = RGB(c); well.color = c
        for s in swatches { s.selected = s.color == c }
        window.makeFirstResponder(hair)
    }

    @objc func toolPicked() { hair.tool = Tool(rawValue: tools.selectedSegment) ?? .brush; window.makeFirstResponder(hair) }
    @objc func baldToggled() { hair.bald = baldSwitch.state == .on; window.makeFirstResponder(hair) }
    @objc func toggleBald() { hair.bald.toggle(); baldSwitch.state = hair.bald ? .on : .off }
    @objc func sizeChanged(_ s: NSSlider) { hair.radius = CGFloat(s.doubleValue) }
    @objc func headChanged(_ s: NSSlider) { camera.hairVolume = Float(s.doubleValue) }
    @objc func strengthChanged(_ s: NSSlider) { camera.strength = Float(s.doubleValue) }
    @objc func wellChanged() { hair.color = RGB(well.color); for s in swatches { s.selected = false } }
    @objc func showCapSettings(_ b: NSButton) {
        if capPopover.isShown { capPopover.close() } else { capPopover.show(relativeTo: b.bounds, of: b, preferredEdge: .maxY) }
    }
    @objc func presetPicked(_ p: NSPopUpButton) {
        if p.indexOfSelectedItem > 0, let t = p.titleOfSelectedItem { hair.applyPreset(t) }
        window.makeFirstResponder(hair)
    }
    @objc func growHair() { hair.growAll([.scalp]) }
    @objc func growBeard() { hair.growAll([.beard, .mustache]) }
    @objc func shaveAll() { hair.shaveAll() }
    @objc func resetStyle() { hair.resetStyle() }
    @objc func recalibrate() { capPopover.close(); hair.recalibrate(); window.makeFirstResponder(hair) }
    @objc func snapshot() { hair.snapshot(); window.makeFirstResponder(hair) }
    /// Saves 10 s of raw camera video (no effects) so tracking can be tuned offline by replaying it.
    @objc func recordClip() {
        capPopover.close()
        let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents/hair game/Clips")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        camera.startRecording(to: dir.appendingPathComponent("Clip \(f.string(from: Date())).mov"))
        hair.toast = ("🎬 Recording 10 s test clip — turn, nod, lean in…", CACurrentMediaTime() + 10)
    }

    private func buildMenu() {
        let main = NSMenu()
        func menu(_ title: String, _ items: [NSMenuItem]) -> NSMenu {
            let item = NSMenuItem(); main.addItem(item)
            let m = NSMenu(title: title); items.forEach(m.addItem); item.submenu = m
            return m
        }
        func item(_ t: String, _ a: Selector, _ key: String, _ mods: NSEvent.ModifierFlags = .command, target: AnyObject? = nil) -> NSMenuItem {
            let i = NSMenuItem(title: t, action: a, keyEquivalent: key); i.keyEquivalentModifierMask = mods; i.target = target
            return i
        }
        _ = menu("Hair Game", [
            item("About Hair Game", #selector(NSApplication.orderFrontStandardAboutPanel(_:)), ""), .separator(),
            item("Hide Hair Game", #selector(NSApplication.hide(_:)), "h"),
            item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", [.command, .option]),
            item("Show All", #selector(NSApplication.unhideAllApplications(_:)), ""), .separator(),
            item("Quit Hair Game", #selector(NSApplication.terminate(_:)), "q")])
        _ = menu("File", [
            item("Save Snapshot", #selector(snapshot), "s", target: self),
            item("Record 10-Second Test Clip", #selector(recordClip), "r", target: self), .separator(),
            item("Close Window", #selector(NSWindow.performClose(_:)), "w")])
        _ = menu("Hair", [
            item("Grow Hair", #selector(growHair), "g", target: self),
            item("Grow Beard", #selector(growBeard), "g", [.command, .shift], target: self),
            item("Shave All", #selector(shaveAll), "x", [.command, .shift], target: self),
            item("Reset Style", #selector(resetStyle), "", target: self), .separator(),
            item("Toggle Bald Cap", #selector(toggleBald), "b", target: self),
            item("Recalibrate Face", #selector(recalibrate), "k", target: self)])
        let win = menu("Window", [
            item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"),
            item("Zoom", #selector(NSWindow.performZoom(_:)), ""),
            item("Enter Full Screen", #selector(NSWindow.toggleFullScreen(_:)), "f", [.command, .control])])
        NSApp.mainMenu = main
        NSApp.windowsMenu = win
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
