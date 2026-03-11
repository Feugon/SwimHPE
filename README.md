# SwimHPE Write-Up
<img width="501" height="570" alt="Screenshot 2026-03-10 at 5 04 19 PM" src="https://github.com/user-attachments/assets/b52a5222-6c6e-47f4-b5a5-8f505570eee8" />


## Project Overview
In this project, I attempt to create a model that is capable of human pose estimation on swimming footage in real time. Such models could be used to detect key statistics about a swimmer like their stroke rate, speed, and angles of their joints, which could aid them in improving their techniques. 

Current SOTA models struggle immensely on this task. Swimming videos are difficult for existing HPE models for several reasons: self-occlusions of joints, occlusions due to water splashing, poor video quality, lack of available training data, and visual warping effects. Here, I mostly attempt to combat the lack of available training data out of these challenges.

By using a mix of synthetic and hand-labeled data, I was able to finetune a YoloPose-26n model to significantly outperform all of its larger variants in the swimming domain.

*AI Use*: Pretty much all of the code written in this codebase was produced by AI. However, the write-up is written by me so you don't have to deal with any more unnecessary AI slop.


## SwimXYZ: Synthetic Dataset
When I first thought about starting this project, I looked into currently available swimming HPE datasets and quickly found SwimXYZ. This dataset consists of 3.4 million annotated frames of synthetically generated swimming footage from different camera angles and of different strokes. As far as I understood it, they purchased a [$19 Unity asset of a swimmer](https://assetstore.unity.com/packages/3d/characters/white-swimmer-10686-tris-39121) and used a GANimator to add some noise to the overall motion. The authors also stated that they "put much effort into rendering a realistic aquatic environment".

Overall, this dataset is not the solution that I was looking for. The authors themselves admit that the data is not diverse at all, the environment and the swimmer being pretty much identical throughout the entire dataset. They claimed to have some sucess in finetuning a VITPose model using the synthetic images, however I am skeptical of their methods. The images they selected were all of aerial view and looked rather high quality, I'm not sure if the base model would've struggled with those images. Additionally, they did give some quantitative measurements but that was done by finetuning the model on freestly synthetic data and then testing on butterfly synthetic data, which I suspect is the model learning the environment and the synthetic swimmer look.

Also, I could not figure out what they were doing with the data they provided. First of all, they use commas for decimal delimiters which I painfully realized after about an hour of pure confusion. Then they have 2D and 3D annotations but they both contain the same amount of values. Why? Then the header for these files include labels that are just not correct, for example the 2D COCO labels have a "MidHip" keypoint which actually doesn't exist and messes up with the importing of the data. It may be that I didn't understand the dataset structure, but if that's the case they did not make it easy. It was a bit of a nightmare to figure out all the abnormalities with the annotations.

I hope I didn't come off as too harsh in the paragraphs above, just wanted to describe the journey that I had while working with this all. While the dataset wasn't enough on its own I found that it was useful as a supplement to real data which I touch upon later. Also I only used freestyle frames in my training (my initial goal was for the model to at least be useful for one stroke), mixing up the strokes should definitely help out a bit more with generalization.


## Hand-Labeling Data
As I quickly realized, training on purely synthetic data was not going to be enough. The data was just not diverse enough and the model could quickly learn to memorize certain poses. To combat this I hand-labeled 420 images of swimming footage I found on YouTube. I attempted to do this only on Creative Common licenses videos at first, but hit a wall with footage that I could find so I had to use non CC videos. In the code there is an entire pipeline of using `yt-dlp` to download the videos and separate them into individual frames.

After the frames were created, I used [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) to manually label them. I did not label any facial features, they were often too occluded and weren't of interest for any downstream tasks for the model. The software was really nice to use and while I did have some complaints I will not list them here because it was probably due to my lack of experience with using it. It took me about 5 hours to find and label the frames.

**Important:** Some of the annotations are not of the highest quality or are outright poor. I didn't know how to handle labeling multiple swimmers in the frame at first. Also I was much too liberal with not marking points because of occlusions when I first started to label points. The dataset is not stored directly in the repo, instead we have a JSON of annotations and a script called `reconstruct_dataset.py` that gets the relevant images and labels for you. Feel free to use this however you please. 

Here are the YT videos I used to gather the frames from:
[1](https://www.youtube.com/watch?v=omhYbRTOQEk)
[2](https://www.youtube.com/watch?v=HNav_fzh0vI)
[3](https://www.youtube.com/watch?v=3gjWQW2ubpA)
[4](https://www.youtube.com/watch?v=gSISVaMTe1o)
[5](https://www.youtube.com/watch?v=fqPwtPwdcnM)
[6](https://www.youtube.com/watch?v=ss7KyDcHrvI)
[7](https://www.youtube.com/watch?v=nrdDanWxQQY)
[8](https://www.youtube.com/watch?v=3gimynmehxA)
[9](https://www.youtube.com/watch?v=bpuSkY-WwYk)
[10](https://www.youtube.com/watch?v=m3BsRGK9RSQ)
[11](https://www.youtube.com/watch?v=wLLbvett15o)
[12](https://www.youtube.com/watch?v=ljGppH9qgBA)
[13](https://www.youtube.com/watch?v=_SjuNR8W3Zc)
[14](https://www.youtube.com/watch?v=7UqIlG1sMNs)
[15](https://www.youtube.com/watch?v=9F_qz4FZZXk)


## Training The Model

In this section, I will describe a few of my experiments and predicaments that I had while figuring out how to best train the model.

### Determining The Synthetic-to-Real Data Ratio
I quickly realized that using all synthetic data did not work at all. But did this mean that the data was totally useless to me? I ran a quick experiment where I finetuned a nano model on a dataset with different synthetic-to-real image ratios. I used 4 different datasets with the following synthetic:real counts: 128:0, 128:64, 32:64, 0:64. What I found was that the 2:1 and the 1:2 ratios midly outperformed the real-only dataset. The difference was slight and my validation set was rather small (I didn't have many real images to work with at the time), but it was enough to convince me to keep some of the synthetic data in the training runs.

Later, as I gathered more hand-labeled data, I turned to a 1:1 ratio. This was chosen by running several quick runs and seeing how the model performs on the validation set. It seems intuitive to me that as the number of real samples increase the need for synthetic ones decrease and hence I could afford to lower the ratio a bit.

### Choosing Number of Layers to Freeze and Model Size
Honestly, this is something that I will come back to. If you look at the Yolo models than you will see that the first 9 layers act as a backbone that extract features from the image. Layers 9-22 make up the neck of the model which refine the features from the backbone. And then there are 3 head blocks that are each tasked with identifying different sized objects in the picture. [source](https://arxiv.org/abs/2602.14582)

I ran quick experiments where I chose to freeze the backbone, backbone and half of neck, and neck+backbone. For the nano model I found that freezing the backbone only yielded the best results, which agrees with the intuition that I had. As I write this, I realize that I didn't ever attempt to finetune the entire model, which I imagine would lead to overfitting but who knows?

I did attempt to finetune a larger model and no matter what I tried the performance was always pretty much the same as the nano model. This really puzzles me and I still have yet to understand why this is happening. Perhaps I just wasn't running the training run for long enough. Like I said, this is an area where I want to continue experimenting within this project. 

### Augmentations
For data augmentations I didn't really know what I was looking for. Obviously, I wanted to diversify the data a bit but I wasn't sure what I was looking for. I have never really done a serious CV project so all of this was completely new to me. I kind of visually checked the results of the augmentations that I performed and made sure that the images were still realistic. This is another area to explore when I come back to the project.

## Inference and Results

After finetuning the nano YoloPose model for about 200 epochs (it automatically cut off at 263 since there was no improvement in performance for a 100 epochs) we got a pretty capable model. Most of my remarks are from a qualitative standpoint, I viewed the performance of the model on an unseen video and judged that. On my test set of 33 images from 2 sources that do no appear in the training I had the following measurements:

| Model | Params | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| yolo26n | 2.9M | 0.182 | 0.160 | 0.050 |
| yolo26s | 10.4M | 0.333 | 0.324 | 0.169 |
| yolo26m | 21.5M | 0.485 | 0.488 | 0.250 |
| yolo26l | 25.9M | 0.506 | 0.523 | 0.312 |
| yolo26x | 57.6M | 0.576 | 0.561 | 0.367 |
| finetuned yolo26n | 2.9M | 0.515 | 0.541 | 0.193 |

And while the measurements are pretty good, I actually think the model is much more capable than the numbers suggest. My validation set is rather small (I didn't want to lose any more data for training) and a qualitative comparison of the models is actually more telling.

### Finetuned Model 


https://github.com/user-attachments/assets/0bd0f7b1-a15c-4823-abbc-28a22c3184c5

### Largest base YoloPose Model 



https://github.com/user-attachments/assets/4447c518-8ddc-493d-86b4-f4760010ee65

As we can see the performance is very similar despite the latter being almost 20x as big. The respective FPS were 25 and 1. 

Additionally I added a simple test time augmentation (TTA) which runs inference on a horizontally flipped version of the image and averages out the predictions. It slows down by a little over 2x but does create much more stable predictions and helps the recall of the model significantly (reducing the empty guesses).

The fine tuned model runs at about 66 FPS on my M1 Macbook when using `.mlpackage` and at about 25 FPS when we have TTA enabled. This is good enough for any real time analysis. I briefly looked into this with an AI, but apparently modern Iphones have faster chips than my M1 Mac and inference could actually run at about 150 FPS there. Not sure how much I trust this, but the point stands that this could be used in a swim practice environment to get real time feedback.


## Future Work
There are several things that I want to try to do in the future to increase the performance of the model:
- Try using RTMPose as the model architecture (might not be real time but should be more accurate)
- Run deeper experiments into synthetic:real dataset ratios
- Experiment with different augmentations
- Create a usable UI to estimate stroke rate, speed, and other relevant metrics
- Create an agentic research playground [Karpathy style](https://x.com/karpathy/status/2031135152349524125)


## Acknowledgments
- [Ultralytics](https://github.com/ultralytics/ultralytics): smooth API for downloading and training models (though I've heard some of the business practices are *questionable*)
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling): great program for hand-labeling
- [SwimXYZ](https://g-fiche.github.io/research-pages/swimxyz/): synthetic dataset
- [YoloPose Paper](https://arxiv.org/abs/2204.06806): introduction to the model I finetuned in this project

### Relevant Papers
Here is a list of papers that I glanced over that aim to create better swim HPE models:
- [Optimizing Human Pose Estimation for Automatic
Analysis of Competitive Swimmers.](https://libstore.ugent.be/fulltxt/RUG01/002/945/759/RUG01-002945759_2021_0001_AC.pdf) - One of the few papers that did something novel. Uses the fact that swimming is cyclical in nature to establish an algorithm that matches current images to anchor poses.
- [SwimmerNET: Underwater 2D Swimmer Pose Estimation Exploiting Fully Convolutional Neural Networks](https://www.mdpi.com/1424-8220/23/4/2364) - Got really good results but trained 8 large, separate models to run all at once. Not suitable for real-time inference
- [Pose estimation for swimmers in video surveillancea](https://link.springer.com/article/10.1007/s11042-023-16618-w) - Most promising model, they modified the HRnet architecture and achieved really good results (though they only used videos with really clean camera angles which may not generalize that well). Couldn't find the model they created or the dataset they labeled (*sigh). Also this is likely not suitable for real-time inference either.
