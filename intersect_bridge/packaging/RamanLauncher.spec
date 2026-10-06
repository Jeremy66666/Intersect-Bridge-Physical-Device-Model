# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['../launcher.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# Windows 10 supplies UCRT/API sets. Do not bundle unrelated libheif runtime
# DLLs discovered through the development environment's PATH.
a.binaries = [entry for entry in a.binaries
              if not entry[0].lower().split('\\')[-1].startswith(('api-ms-win-', 'ext-ms-win-'))
              and entry[0].lower().split('\\')[-1] not in ('ucrtbase.dll', 'icuuc.dll', 'icudt78.dll')]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RamanLauncher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
