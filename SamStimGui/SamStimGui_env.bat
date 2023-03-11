call conda create --name SamStimGui python=3 --yes
call activate SamStimGui
call python -m pip install PyQt5==5.15.6 --index-url https://pypi.org/simple
call python -m pip install pandas==1.4.2 --index-url https://pypi.org/simple
call python -m pip install openpyxl==3.0.10 --index-url https://pypi.org/simple
cmd /k