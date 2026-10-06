# Active Raman Autofocus project

The active motor source is now:
`C:\Users\Jeremy666\Downloads\portableRamanSpec_Software (1)\RamanAutofocus\RamanAutofocus\RamanAutofocusTest.cpp`

Open the existing project:
`C:\Users\Jeremy666\Downloads\portableRamanSpec_Software (1)\RamanAutofocus\CMakeLists.txt`

The existing MSVC2019 Qt Creator build directory and `.user` configuration are retained. Qt now launches:
`C:\Users\Jeremy666\Downloads\portableRamanSpec_Software (1)\RamanAutofocus\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\RamanAutofocus\RamanAutofocus.exe`

Edit this source and build the RamanAutofocus target in that configuration, or run `rebuild_raman_autofocus.cmd` from the laser_test folder. Stop acquisition and close the controller before building. Start scans from laser_test, not by running the motor executable directly: it needs session arguments and Qt acknowledgements.

The new handshake and X/Y 1D mode were migrated with scan_handshake.h and line_scan_plan.h. Existing CMake project files, SDK paths, unrelated source files and SampleApl were not modified. The previous local hardware/motor folder is retained but no longer used by the production Qt controller. Test emulators still use their isolated local paths.

The user's Z offset 2.10 mm was preserved. The legacy 2D sequence homes Z, moves to 20 mm, returns to 2.10 mm, acquires at 2.20, 2.30, 2.40 and 2.50 mm, then uses 2.30 mm for XY. It does NOT select focus from spectral peaks. Confirm physical clearance before using this sequence. 1D leaves Z unchanged. Stop does not turn off the laser.

Backup before migration:
`C:\Users\Jeremy666\.codex\visualizations\2026\08\26\01a03fa9-2a21-7b23-875f-a65ea5fd7fc2\backup_before_raman_migration_20260922`

No old source, executable or measurement data was deleted. No physical device was started during migration.

Verification: existing Downloads motor target and production Qt compiled successfully. Full hardware-free handshake_tests suite passed, including 1D X/Y, pause/resume, failures, dark storage and focus preview. The 120 handwritten XY calls match the user's previous source in order and coordinates. Both handshake headers match the previously working versions. The `.user` configuration hash is unchanged. Motor compilation has an existing-style C4819 encoding warning for comments; no compile/link errors. Actual hardware movement and focus at the preserved 2.10 mm offset have not been tested.
