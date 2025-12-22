# How to Run Training Script

## Problem: Path with Spaces Error

If you're getting an error like:
```
E:\CSE : The term 'E:\CSE' is not recognized...
```

This happens because PowerShell has trouble with paths containing spaces.

## Solutions

### Option 1: Use the Helper Scripts (Recommended)

#### For PowerShell:
```powershell
.\run_training.ps1
```

#### For Command Prompt:
```cmd
run_training.bat
```

These scripts automatically:
- Handle paths with spaces correctly
- Activate the virtual environment
- Run the training script

### Option 2: Manual Run (PowerShell)

1. **Navigate to the project directory:**
```powershell
cd "E:\CSE Program\Ball Sport Detector Project\Ball-Sport-detector"
```

2. **Activate virtual environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

3. **Run the script:**
```powershell
python training_model.py
```

### Option 3: Manual Run (Command Prompt)

1. **Navigate to the project directory:**
```cmd
cd /d "E:\CSE Program\Ball Sport Detector Project\Ball-Sport-detector"
```

2. **Activate virtual environment:**
```cmd
venv\Scripts\activate.bat
```

3. **Run the script:**
```cmd
python training_model.py
```

### Option 4: Use Python Directly

If you have Python in your PATH, you can run:
```powershell
python "E:\CSE Program\Ball Sport Detector Project\Ball-Sport-detector\training_model.py"
```

## Important Notes

1. **Always quote paths with spaces** in PowerShell/Command Prompt
2. **Activate the virtual environment first** before running the script
3. **Update dataset paths** in `training_model.py` if your dataset is in a different location

## Troubleshooting

### If virtual environment activation fails:
- Make sure you're in the correct directory
- Check that `venv` or `.venv` folder exists
- Try: `python -m venv venv` to recreate it

### If you get import errors:
- Make sure virtual environment is activated (you should see `(venv)` in your prompt)
- Install dependencies: `pip install -r requirements.txt`

### If dataset path errors occur:
- Update `TRAIN_IMAGE` and `TEST_IMAGE` paths in `training_model.py`
- Make sure the paths point to folders containing your ball class subdirectories

