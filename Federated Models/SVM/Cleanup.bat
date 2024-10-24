@echo off
set /p n=Enter the number of copies to clean up: 

for /L %%i in (1, 1, %n%) do (
	if exist node%%i.py (
		del node%%i.py
		echo node%%i.py deleted
	) else (
		echo node%%i.py does not exist
	)
)

echo Cleanup completed.
pause
