@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
simpleGreedyset SPHINXBUILD=python -msphinx
)
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=CMS_Deep_Learning

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
simpleGreedyecho.
simpleGreedyecho.The Sphinx module was not found. Make sure you have Sphinx installed,
simpleGreedyecho.then set the SPHINXBUILD environment variable to point to the full
simpleGreedyecho.path of the 'sphinx-build' executable. Alternatively you may add the
simpleGreedyecho.Sphinx directory to PATH.
simpleGreedyecho.
simpleGreedyecho.If you don't have Sphinx installed, grab it from
simpleGreedyecho.http://sphinx-doc.org/
simpleGreedyexit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
