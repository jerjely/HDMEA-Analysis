; Inno Setup script — per-user install (no admin rights), Start menu entry, optional desktop shortcut.
; Built by the GitHub workflow:
;   iscc /DAppVersion=1.2.3 /DNumVersion=1.2.3 /DSourceDir=dist\HDMEA-Analysis /DOutputDir=release installer\windows\HDMEA-Analysis.iss
#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef NumVersion
  #define NumVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\HDMEA-Analysis"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\release"
#endif
#define AppName "HDMEA-Analysis"
#define AppTitle "HD-MEA Analysis"
#define AppExe "HDMEA-Analysis.exe"

[Setup]
AppId={{6C1F7B2E-4F7A-4E55-9C77-2B4D9A1E8D31}
AppName={#AppTitle}
AppVersion={#AppVersion}
AppVerName={#AppTitle} {#AppVersion}
AppPublisher=Gergely Szarka
AppPublisherURL=https://github.com/jerjely/HDMEA-Analysis
AppSupportURL=https://github.com/jerjely/HDMEA-Analysis/issues
VersionInfoVersion={#NumVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppTitle}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename={#AppName}-{#AppVersion}-Windows-setup
SetupIconFile=..\..\hdmea\resources\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppTitle}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppTitle}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppTitle}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppTitle}}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; remove files of an older version before copying the new one
Type: filesandordirs; Name: "{app}\_internal"
