# Ethical Engineers Project

### Demonstrating Racial Bias in Facial Recognition

https://github.com/DevTechS/EthicalEngineersProject

Dataset Link
https://github.com/joojs/fairface

In order to run these scripts you may need to install some libraries. Most notably OpenCV, face_recognition, numpy. Additionally you will need to download the full dataset separately, because Github limits how many files I can upload to the repository. On my PC Each script takes about two hours to run. There's progress updates about once every minute, but it still takes a very long time. Speed was beyond the scope of this project.

The goal of this project was to demonstrate the racial bias in various facial recognition algorithms. For this we used the fairface dataset which has pictures of over 90,000 faces, along with age, race, and gender information. The ideal facial recognition system is both fair and accurate, detecting more faces is good, but not singling out any particular group is also important.

## OpenCV

First, and most simple to use is OpenCV's CascadeClassifier. This library is free, open source and very easy to run, relying on computational image processing to find the face, eyes, and mouth of a subject. The `Annotate_cv2.py` script uses this library, takes the fairface training data as an input, and renders all this out into the output folder `done_cv2_train` and the csv `output_cv2.csv`. The script also annotates eyes and mouths, but the scoring will just be based on face detection. Finally I used `output_scoring.py` to show the final scores between all the races. Which are as follows:

```
White: 57.4%
East Asian: 63.0%
Middle Eastern: 56.0%
Black: 48.6%
Indian: 63.3%
Latino_Hispanic: 63.0%
Southeast Asian: 64.0%
Overall: 59.3%
```

This reveals that fairface is quite a difficult dataset, with more than a third of the faces going unrecognized. It does the worst with Black skin tones, and does the best with Southeast Asian. This means that the worst group has 42.8% more error than the best. Fortunately there is

## face_recognition

This code, found in `Annotate_face_recognition.py`, works much the same way as the last approach, but uses newer machine learning techniques to achieve a better result. The overall rates are here:

```
White: 71.2%
East Asian: 75.4%
Middle Eastern: 70.1%
Black: 69.1%
Indian: 76.1%
Latino_Hispanic: 76.7%
Southeast Asian: 77.0%
Overall: 73.7%
```

As you can see this is a significant improvement, with 35.4% fewer errors overall. In terms of fairness the worst and best groups are Black and Southeast Asian once again. This time the worse group is 34.3% more error prone than the best. This is a small improvement, but an improvement none the less!

## Custom Algorithm

So using this information, I decided to see if I can augment these tools with some preprocessing. I tried many things, but it turns out many image enhancements like edge filters and contrast adjustments are actually surprisingly counter productive. I settled on an inverse gaussian with a sigma of 10, resulting in a sharpness filter, as well as a simple brightness normalization system. Then, I used face_recognition. And in case that doesn't work, then I use OpenCV's CascadeClassifier as a fallback. All that resulted in these scores:

```
White: 76.2%
East Asian: 80.5%
Middle Eastern: 75.1%
Black: 74.6%
Indian: 80.1%
Latino_Hispanic: 80.3%
Southeast Asian: 81.6%
Overall: 78.3%
```

Overall, its a decent although not groundbreaking improvement, with 24.3% fewer errors than face_recognition alone. Once again Black and Southeast Asian are the worst and best groups, leaving us with a 38.0% difference in errors between them. This is a problem because my improvements, though overall beneficial, benefited some groups more than others, increasing the bias of the system.

## Conclusion

Going into this I believed the bias in these system was due to engineers focusing on other aspects such as accuracy, performance, and time to deployment. But this project has demonstrated that this is a very difficult problem with no easy solution. I was able to show that AI based systems are superior to traditional methods, and that newer techniques tend to be more fair, so we're moving in the right direction, but we still have a long way to go.

As an extension to this project, perhaps in the future we can train an AI on a specially curated dataset to be as fair and inclusive as possible, Its accuracy will likely not be as good as the state of the art, but if we can prove that fairness is possible with an AI system, then perhaps it can lead to further development in the future.
