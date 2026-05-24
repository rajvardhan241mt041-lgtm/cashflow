Write-Host "Installing API requirements..."
pip install -r api\requirements.txt
Write-Host "Installing Worker requirements..."
pip install -r worker\requirements.txt
Write-Host "Installing UI requirements..."
pip install -r ui\requirements.txt

Write-Host "Starting API in a new window..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit -Command `"python -m uvicorn main:app --host 0.0.0.0 --port 8000`"" -WorkingDirectory "$PWD\api"

Write-Host "Starting Worker in a new window..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit -Command `"python nightly_pipeline.py`"" -WorkingDirectory "$PWD\worker"

Write-Host "Waiting for API to start..."
Start-Sleep -Seconds 5

Write-Host "Starting UI in a new window..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit -Command `"`$env:API_URL='http://localhost:8000'; python -m streamlit run dashboard.py --server.port=8501`"" -WorkingDirectory "$PWD\ui"

Write-Host "App started! The browser should open automatically."
