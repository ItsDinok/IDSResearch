@echo off
set /p n=Enter the number of copies: 

start /wait cmd /c py Quick.py %n%
start cmd /k C:/Users/markd/Desktop/Python311/python.exe Server.py %n%

for /L %%i in (1, 1, %n%) do (
	copy FederatedANN.py node%%i.py
	:: %n% for vertical, -t for transfer, none for horizontal
	start cmd /k C:/Users/markd/Desktop/Python311/python.exe node%%i.py %n%
)

echo %n% copies of node.py have been created and executed.
start cmd /k C:/Users/markd/Desktop/Python311/python.exe ANNHelper.py


