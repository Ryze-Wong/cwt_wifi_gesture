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