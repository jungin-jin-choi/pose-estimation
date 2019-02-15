
## 3. Code for Single Human Pose Estimation in OpenCV

일단은 Single Person Pose Estimation부터 시작하자.

## 3.1. Step 1 : Download Model Weights

다음을 순차적으로 실행한다.
```
git clone https://github.com/spmallick/learnopencv.git
cd learnopencv/OpenPose
sudo chmod a+x getModels.sh
./getModels.sh
```
다운로드 과정을 성공적으로 실행했다면, 현재 디렉토리 (learnopencv/OpenPose) 내에 pose, face, hand 디렉토리가 생성되어 있을 것이고, 그 안에 아래 파일들을 확인할 수 있을 것이다.
```
pose/coco/pose_iter_440000.caffemodel
pose/mpi/pose_iter_160000.caffemodel
face/pose_iter_116000.caffemodel
hand/pose_iter_102000.caffemodel
```

참고로 위 모델들은 size가 매우 크기 때문에, github 상에 push되지 않는다. 이 점 참고하고 로컬에서 돌려보시길.

## 3.2 Step 2: Load Network

아래 코드를 실행하면  .prototxt 파일과 .caffemodel 파일을 볼 수 있다.
```
cd pose/coo
ls
```
.prototxt 파일은 뉴럴 네트워크의 아키텍처를 정의하고, .caffemodel 파일은 trained model의 weights들을 저장한다. 

이제 Jupyter Notebook에서 본 repository 상의 Korean_Tutorial_OpenPose.ipynb 파일을 열자.
해당 파일의 영문 원본은 learnopencv/OpenPose/OpenPose_Notebook.ipynb이다.

Step 2, Step 3, Step 4의 자세한 설명은 Korean_Tutorial_OpenPose.ipynb에서 볼 수 있다.
본 README.md에서는 코드의 큰 흐름만을 짚고 넘어갈 것이다.

정리하자면, 현재 디렉토리 내에 아래와 같은 파일 및 디렉토리들이 존재해야 한다.
```
CMakeLists.txt
face
getModels.sh
hand
img
Korean_Tutorial_OpenPose.ipynb
learnopencv
multiple.jpeg
OpenPoseImage.cpp
OpenPoseImage.py
OpenPoseVideo.cpp
OpenPoseVideo.py
pose
README
sample_video.mp4
single.jpeg
```
face, hand, pose 디렉토리 내에 caffemodel들이 있어야 하고,
현재 디렉토리에서 Korean_Tutorial_OpenPose.ipynb를 열고 실행시키면 된다. 

## 3.3. Step 3: Read Image and Prepare Input to the Network

## 3.4. Step 4: Make Predictions and Parse Keypoints

## 3.5. Step 5: Draw Skeleton
