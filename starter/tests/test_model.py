import sys
sys.path.insert(1, 'starter/starter/ml/model.py')
from model import train_model, write_model

def test_model_creation():
    X_data = None
    Y_data = None
    model = train_model(X_data, Y_data)
    assert model != None

def test_write_model():
    model = None
    dest_pth = 'test_data'
    write_model(model, dest_pth)
    
def run_all():
    test_model_creation()
    test_write_model()