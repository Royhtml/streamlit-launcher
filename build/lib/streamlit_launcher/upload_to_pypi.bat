@echo off
echo ==============================
echo   Build & Upload to PyPI
echo ==============================

REM --- SET ENVIRONMENT VARIABLES ---
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcCJGYyZWZkNjU0LWU1NzctNDQ5ZC1hOWYwLTMzYzlhODFiZTJjNwACKlszLCIyMjI2NTRkZi01NDFmLTRhYWEtOWQzMC1hNjk0MmZiN2ZjMzQiXQAABiA-mbtKH-zfWugSn0lPU2vtIv-T6WPSAVatDhglvn3M3A
REM --- HAPUS FOLDER DIST SEBELUM BUILD ---
if exist dist rmdir /s /q dist

REM --- BUILD PACKAGE (SDIST & WHEEL) ---
python setup.py sdist bdist_wheel

REM --- UPLOAD KE PYPI ---
twine upload dist/*

pause
