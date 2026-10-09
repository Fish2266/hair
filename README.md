# Hair Game

A lightweight macOS app that uses your webcam and Apple's Vision face tracking to make you bald, then lets you grow, brush, cut, curl and dye physics-simulated hair and facial hair.

- **Bald cap:** Vision person segmentation + face landmarks find your real hair, which is replaced by a shaded scalp and a learned background. It follows head turns, nods and leaning in.
- **Hair:** 3D guide strands (Verlet + follow-the-leader) around a fitted skull, rendered with Metal as clumps of tapered, anti-aliased ribbons.
- **Tools:** brush, growth serum, razor, scissors, dye, gel, curler, blow-dryer; hairstyle and facial-hair presets; snapshots.

## Build

Requires macOS 14+ and the Xcode command-line tools.

```bash
./build.sh
open "Hair Game.app"
```

## Keys

`1`–`8` tools · `B` bald cap · scroll = brush size · `Space` snapshot · `⌘R` record a 10-second test clip (saved to `Clips/`, which is git-ignored)
