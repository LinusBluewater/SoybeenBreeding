@echo off

REM ===== 最先保存脚本目录 =====
set SCRIPT_DIR=%~dp0

SET xmx=-Xmx1536m
SET xms=-Xms512m
SET args=
set space= 

:top
SET found=0

ECHO %1 | FINDSTR /C:"-Xmx" >nul & IF NOT ERRORLEVEL 1 (
    SET xmx=%1
    SET found=1
)

ECHO %1 | FINDSTR /C:"-Xms" >nul & IF NOT ERRORLEVEL 1 (
    SET xms=%1
    SET found=1
)

IF %found%==0 SET args=%args%%space%%1

SHIFT
IF NOT [%1]==[] GOTO top

REM ===== 使用保存好的路径 =====
set TOP=%SCRIPT_DIR%

set LIB_JARS=%TOP%lib

set CP=%TOP%sTASSEL.jar

for %%i in (%LIB_JARS%\*.jar) do call "%TOP%cp.bat" %%i

echo %CP%

java -classpath "%CP%" %xms% %xmx% net.maizegenetics.pipeline.TasselPipeline %args%