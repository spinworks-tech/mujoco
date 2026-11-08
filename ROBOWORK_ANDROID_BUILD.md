# MuJoCo for Android ARM64 (RoboWork)

RoboWork's `RoboSim/Assets/RoboWork/Plugins/Android/arm64-v8a/libmujoco.so` is a Release
build of official MuJoCo 4.0.0, tag commit `ac60eae30fa7e0d31315c38d9476752fb81be67e`.
The source is unmodified. It matches the `org.mujoco` 4.0.0 Unity bindings
(`unity/` in this repository), which import `mujoco`.

- Minimum Android API: **28 / Android 9**. Upstream uses `aligned_alloc`, which
  Android introduced at API 28. PlayerSettings uses the same minimum.
- ABI: `arm64-v8a`. Unity imports the library for Android ARM64 only.
- Toolchain: Unity 6000.6.2f1's Android NDK r27c / 27.2.12479018, Clang 18.0.3.
- Release options: `-O3`, `NDEBUG`, thin LTO; debug and unnecessary symbols stripped.
- ELF load segments have 16 KB alignment. Unity reports `Is16KbAligned: true`.
- C++ runtime and MuJoCo dependencies are linked statically. Dynamic dependencies
  are Android's `libc.so`, `libm.so`, and `libdl.so`.
- Size: 5,896,208 bytes.
- SHA-256: `2b097c67f9152797b714c18084f85d9ee4f336f9e3d355fbc00a18820dccbc2c`.

## Rebuild on Windows

Run from the RoboSim repository root in PowerShell. Set `$mujocoNdk` to Unity's
installed Android NDK directory. Keep every `-D` argument quoted.

```powershell
$mujocoNdk = 'C:/Program Files/Unity/Hub/Editor/6000.6.2f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK'
git clone --depth 1 --branch 4.0.0 https://github.com/google-deepmind/mujoco.git tmp/mujoco-4.0.0-android-source
cmake -S tmp/mujoco-4.0.0-android-source -B tmp/mujoco-4.0.0-android-arm64-build -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$mujocoNdk/build/cmake/android.toolchain.cmake" `
  '-DCMAKE_CXX_FLAGS=--driver-mode=g++' `
  '-DANDROID_ABI=arm64-v8a' '-DANDROID_PLATFORM=android-28' `
  '-DANDROID_STL=c++_static' '-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON' `
  '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_POLICY_VERSION_MINIMUM=3.16' `
  '-DMUJOCO_BUILD_SIMULATE=OFF' '-DMUJOCO_BUILD_EXAMPLES=OFF' `
  '-DMUJOCO_BUILD_TESTS=OFF' '-DMUJOCO_TEST_PYTHON_UTIL=OFF' `
  '-DMUJOCO_ENABLE_AVX=OFF' '-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF'
cmake --build tmp/mujoco-4.0.0-android-arm64-build --target mujoco --parallel 8
& "$mujocoNdk/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe" `
  --strip-unneeded -o tmp/libmujoco-4.0.0-android-arm64.so `
  tmp/mujoco-4.0.0-android-arm64-build/lib/libmujoco.so
```

Explicit C++ driver mode is necessary because this Windows CMake/NDK combination
uses the compiler's short filename when linking. Without the driver option,
compiler detection omits libc++ and the library link fails.

Verify the output before replacing the asset. Import and configure it through
Unity; let Unity generate metadata. Do not patch the MuJoCo source for this build.

## Verification on 2026-11-08

Rebuilt from tag 4.0.0 with the same recipe and options; only the version changed.
Native compilation and linking succeeded. ELF inspection confirmed AArch64, the
`libmujoco.so` SONAME, API 28 (`.note.android.ident`; `aligned_alloc` is still
used, `LIBC_P`), 16 KB load alignment (`0x4000` on all three LOAD segments), and
dynamic dependencies limited to `libc.so`, `libm.so` and `libdl.so`. All 350
P/Invoke function names in the 4.0.0 Unity bindings are exported (821 exports in
all); `mj_version` is exported and `mjVERSION_HEADER` is 4000000.

An Android APK and device execution have not been verified. The project's HDRP
pipeline reports that Android is unsupported, and no Android device is connected.
MuJoCo's license is this repository's `LICENSE`. The notices of the statically
linked dependencies (ccd, lodepng, MarchingCubeCpp, miniz, qhull, tinyobjloader,
tinyxml2) are in the official release's `THIRD_PARTY_NOTICES.txt`; they matched the
4.0.0 sources on 2026-11-08.

Previous build, 3.15.0 (verified 2026-10-07): tag commit
`9ea3cdfcae93bf2cc4dc0e1a1627c5a39a1e06e5`, 5,674,312 bytes, SHA-256
`5767d2ad714a721a85911e9f4c9aebcfa9f42add298ec091bbe22a99de550e40`.
