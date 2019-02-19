# Network Training

0) Download dataset

dataset뿐만 아니라 annotation과 COCO official toolbox도 다운받아야 한다.

1) Preprocessing

터미널 상 현재 디렉토리에서 아래의 코드를 시행한다. 
```python generate_json_mask.py```
위 코드를 통해 COCO 데이터셋으로부터 json mask파일을 생성해낼 수 있다. 

2) Set training parameters

아래 파일에서 하이퍼파라미터들을 설정할 수 있다. 
```config.yml```
batch size, momentum, learning rate 등을 이곳에서 조정할 수 있다.

3) Train the model

터미널 상 현재 디렉토리에서 아래의 코드를 시행한다.
```sh train.sh```
참고로 이는 위에서 preprocessing된 json mask들과 함께 ```train_pose.py```파일을 시행한다.