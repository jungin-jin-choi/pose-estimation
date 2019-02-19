
## 4. Code for Multi Human Pose Estimation in OpenCV

이제 Multi Person Pose Estimation이다.

## 4.1. Step 1 : Download Model Weights

다음을 순차적으로 실행한다.
```
git clone https://github.com/spmallick/learnopencv.git
cd learnopencv/OpenPose-Multi-Person
sudo chmod a+x getModels.sh
./getModels.sh
```
다운로드 과정을 성공적으로 실행했다면, 현재 디렉토리 (learnopencv/OpenPose) 내에 pose 디렉토리가 생성되어 있을 것이고, 그 안에 아래 파일들을 확인할 수 있을 것이다.
```
pose/coco/pose_iter_440000.caffemodel
pose/coco/pose_deploy_linevec.prototxt
```

참고로 위 모델들은 size가 매우 크기 때문에, github 상에 push되지 않는다. 이 점 참고하고 로컬에서 돌려보시길.

## 4.2. Step 2: Generate output from image

### 4.2.1. Load Network

아래 코드를 실행하면  .prototxt 파일과 .caffemodel 파일을 볼 수 있다.
```
cd pose/coo
ls
```
.prototxt 파일은 뉴럴 네트워크의 아키텍처를 정의하고, .caffemodel 파일은 trained model의 weights들을 저장한다. 

이제 Jupyter Notebook에서 본 repository 상의 Multi_Tutorial_Korean.ipynb 파일을 열자.
해당 파일의 영문 원본은 learnopencv/OpenPose-Multi-Person/multi-person-openpose.ipynb이다.

Step 2, Step 3, Step 4의 자세한 설명은 Multi_Tutorial_Korean.ipynb에서 볼 수 있다.
본 README.md에서는 코드의 큰 흐름만을 짚고 넘어갈 것이다.

정리하자면, 현재 디렉토리 내에 아래와 같은 파일 및 디렉토리들이 존재해야 한다.
```
README
pose-estimation-paf-equation.png
pose
multi-person-openpose.py
Multi_Tutorial_Korean.ipynb
multi-person-openpose.cpp
group.jpg
getModels.sh
CMakeLists.txt
```
pose 디렉토리 내에 caffemodel이 있어야 하고,
현재 디렉토리에서 Multi_Tutorial_Korean.ipynb를 열고 실행시키면 된다. 

### 4.2.2. Load Image and create input blob

### 4.2.3. Forward pass through the Net

### 4.2.4. Sample Output

## 4.3. Step 3: Detection of keypoints

## 4.4. Step 4 : Find Valid Pairs

## 4.5. Step 5 : Assemble Person-wise Keypoints

## 5. Results
