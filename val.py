from ultralytics import YOLO

def val():
    model = YOLO('models/yolo26m-pose.pt')  # load a pretrained model (recommended for training)
    results = model.val(data="configs/swimXYZ.yaml", verbose=False)

    print(results)  # print results to console

def predict():
    model = YOLO('models/yolo26n-pose.pt')  # load a pretrained model (recommended for training)
    results = model.predict("examples/ex.jpeg")  

    for r in results:  # r is a Results object
        print(r.boxes)  # Boxes object for bbox outputs
        print(r.keypoints)  # Keypoints object for keypoint outputs
        print(r.masks)  # Masks object for segmentation outputs
        r.show()  # display results

val()