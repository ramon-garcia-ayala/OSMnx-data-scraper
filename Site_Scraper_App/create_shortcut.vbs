Dim WshShell, oShortcut, sDesktop, sBatPath

Set WshShell = CreateObject("WScript.Shell")
sDesktop = WshShell.SpecialFolders("Desktop")
sBatPath = "E:\IAAC Local GIT Repositories\OSMnx-data-scraper\launch_site_scraper.bat"

Set oShortcut = WshShell.CreateShortcut(sDesktop & "\Site Scraper App.lnk")
oShortcut.TargetPath = sBatPath
oShortcut.WorkingDirectory = "E:\IAAC Local GIT Repositories\OSMnx-data-scraper"
oShortcut.Description = "Launch Site Scraper App on localhost:5000"
oShortcut.WindowStyle = 7
oShortcut.Save

MsgBox "Shortcut created on Desktop: 'Site Scraper App'" & Chr(10) & _
       "Double-click it to launch the app and open your browser automatically.", _
       64, "Site Scraper App"
