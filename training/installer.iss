; Inno Setup script for Unsloth Trainer Studio installer
; Requires: Inno Setup 6+

#define MyAppName "Unsloth Trainer Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "YOUR_NAME_OR_ORG"
#define MyAppExeName "UnslothTrainerGUI.exe"

[Setup]
AppId={{9F8D08E9-AB0B-4D74-9E8B-15DE4BFD3A42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE_MIT.txt
OutputBaseFilename=UnslothTrainerStudioSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
