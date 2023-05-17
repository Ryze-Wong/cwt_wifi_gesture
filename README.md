model:
ResNet18
ResNet18_CBAM

dataset:
ARIL
SignFi

python run.py --model ResNet18 --dataset ARIL
python run.py --model ResNet18_CBAM --dataset ARIL
python run.py --model ResNet18 --dataset SignFi
python run.py --model ResNet18_CBAM --dataset SignFi

python run_2.py --model ResNet18 --dataset ARIL
python run_2.py --model ResNet18_CBAM --dataset ARIL
python run_2.py --model ResNet18 --dataset SignFi
python run_2.py --model ResNet18_CBAM --dataset SignFi