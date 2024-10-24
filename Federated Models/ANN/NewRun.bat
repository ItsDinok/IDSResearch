@echo off
set /p n=Enter the number of copies: 

for /L %%i in (1, 1, %n%) do (
	copy node.py node%%i.py
	start cmd /c C:/Users/markd/Desktop/Python311/python.exe node%%i
)

echo %n% copies of node.py have been created and executed.

start cmd /k Server.py
start cmd /k Helper.py

