poetry run jupyter lab --ip $(python3 -c "import subprocess; subprocess.run(['hostname', '-I'], text=True).stdout")
