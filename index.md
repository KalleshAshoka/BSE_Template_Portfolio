<!--- #Object Detection
Replace this text with a brief description (2-3 sentences) of your project. This description should draw the reader in and make them interested in what you've built. You can include what the biggest challenges, takeaways, and triumphs from completing the project were. As you complete your portfolio, remember your audience is less familiar than you are with all that your project entails! --->




| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Kallesh A | Mission San Jose High School | Software Engineering | Incoming Junior


<!--- **Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.** ---> 
<img src="https://raw.githubusercontent.com/KalleshAshoka/Kallesh-Portfolio/gh-pages/Kallesh%20A%20(2).jpg" alt="Headshot Image" height="600px">


  

<h2> Milestone 3</h2>

<iframe width="560" height="315" src="https://www.youtube.com/embed/tuY4uSTVrEc" title="Kallesh A. Third Milestone" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<h4> Summary </h4>

Since my previous milestone I have been able to incorporate my code along with Tensor Flow to detect objects and get my project working properly. I cloned a GitHub repository so that is where I got my data set from and now I can detect a variety of things from house hold items, electronics and and animals. Key topics I learned about here at BSE was I learned how TensorFlow worked, how Raspberry Pi’s work and how HTML code works, and ways to train your model using Teachable Machine and how to maneuver through Github and use open source code.

<h4> Challenges </h4>

My biggest challenges was that I had an error that was very difficult to solve and only worked after completely running all the code back by deleting it all and then had to run all the code commands again. My biggest triumph was when I fixed this issue and finally got my project to work.

<h4> Future Plans </h4>

In the future I hope to learn more about different types of coding languages and more hardware aspects as my project didn’t rely much on the hardware aspect and more software so I would like to learn more about the hardware aspect of things. 




<h2> Milestone 2 </h2>

<iframe width="560" height="315" src="https://www.youtube.com/embed/84xeKC2HaYY" title="Kallesh A. Second Milestone" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<h4> Summary </h4>

Since my previous milestone I have added a camera and the specific camera I added is the Raspberry Pi Camera Module 3. Then I had to test this camera with the code “libcamera-hello -t 0”. This line of code is entered into the terminal on the Pi and it causes the camera to turn on for an indefinite amount of time and you can turn of the camera by pressing “ctr C” in the same terminal as the run command.  The camera contributes to the final goal as it is fundamental for the object detection and can’t be done without it.

<h4> Challenges </h4>

The thing that was surprising about the project was that I thought it would be more challenging but everything seems to be running smooth for the most part. But I think I will face some issues as for the 3rd milestone it will be more coding intensive and I can run into some issues there. Previous challenges I faced was the Adafruit BrainCraft HAT not working so I decided to use my monitor instead as it would be easier and higher resolution and overall easier to use. 

<h4> Future Plans </h4> 
Some things that need to be completed before my final milestone are that I need to get the code for the actual object detection and get it working and set up the data set and do some AI training on parts to get it working properly. 



<h2>Milestone 1</h2>

<iframe width="560" height="315" src="https://www.youtube.com/embed/7pln_3EW9Eg" title="Kallesh A. First Milestone" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<h4> Summary </h4>
My project is a custom object detection system. It uses TensorFlow Lite with a Raspberry Pi 4 and the BrainCraft HAT. It will be able to detect and recognize specific objects and has many real life uses such as it can be used in robotics and security systems. The main components needed for this project are a Raspberry Pi and it runs software to process data. The Adafruit BrainCraft HAT, is connected to the Raspberry Pi and provides a display or you can use a monitor instead with I have done. A keyboard and mouse to control the Raspberry Pi. And along with that to capture video I am using a  Raspberry Pi Camera Module 3. This video can be shown on both the Adafruit HAT or a laptop/monitor. The laptop uses OBS and manages the video output by the camera. And you would need a power supply to supply the entire project and allow it to run. Progress I have made so far is that I have attached all my hardware together and updated my Pi and have it all set up for my next milestones. 

<h4> Challenges </h4>
Challenges I am facing was getting output through the BrainCraft HAT when the monitor was working along with it so to bypass this I am just going to use the computer screen through OBS as I have a bigger screen and better resolution this way and is easier to work with the rest of the project is working smoothly with no issues. 


<h4> Future Plans </h4>
My plan to complete my project is to first test all my compnents so next I will test my camera and fix issues if there are any and then after that do milestone 2 wich would be getting a working camera and then finally run the code and get the entire project working wich would be my milestone 3 and finally work on my modifications. 



<h2> Schematics </h2>
<img src="https://raw.githubusercontent.com/KalleshAshoka/Kallesh-Portfolio/gh-pages/pi-4-to-cam-connection-1024x368.webp" alt="Schematic Image" height = "300px"> 

 Image sourced from: https://cheaperonline.onlinefactory2024.ru/category?name=raspberry%20pi%204%20kamera 



<h2> Code </h2>

```python

# SPDX-FileCopyrightText: 2019 Leigh Johnson
#
# SPDX-License-Identifier: MIT

# Example here:
# https://www.tensorflow.org/beta/tutorials/load_data/images#setup

import tensorflow as tf
import pathlib
import random
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
AUTOTUNE = tf.data.experimental.AUTOTUNE


class FlowerDataset(object):
    def __init__(
        self,
        image_size=(192, 192),
        image_channels=3
    ):
        self.image_size = image_size
        self.image_channels = image_channels

        self.image_label_ds = self.init_dataset()

    def init_dataset(self):
        data_root = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                            fname='flower_photos', untar=True)
        data_root = pathlib.Path(data_root)

        image_paths = list(data_root.glob('*/*'))
        image_paths = [str(path) for path in image_paths]
        random.shuffle(image_paths)
        self.image_paths = image_paths
        label_names = sorted(
            item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index)
                              for index, name in enumerate(label_names))
        self.label_names = label_names

        image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in image_paths]

        return self.init_tensors(image_paths, image_labels)

    def init_tensors(self, image_paths, image_labels):
        ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
        image_label_ds = ds.map(self.load_and_preprocess_from_path_label)
        return image_label_ds

    def load_and_preprocess_from_path_label(self, path, label):
        return self.load_and_preprocess_image(path), label

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.image_channels)
        image = tf.image.resize(image, self.image_size)
        # image /= 255.0  # normalize to [0,1] range
        # image = 2*image-1  # normalize to [-1, 1] range
        image = preprocess_input(image)
        return image
```

<h2> Bill of Materials </h2>

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| CanaKit Raspberry Pi 4 Starter Kit | Image/Object Detection | $119.95 | <a href="https://www.canakit.com/raspberry-pi-4-starter-kit.html/"> Link </a> |
| Logitech K120 Keyboard | Input | $12.34 | <a href="https://www.amazon.com/Logitech-920-002478-K120-USB-Keyboard/dp/B003ELVLKU/ref=sr_1_2?crid=3QE83NKLCZ1UB&dib=eyJ2IjoiMSJ9.hGINAjjbAmcnMmhSu62W7ybtCHaT8ifr068BE_xt70sqaJKSERXvtm9l4hcYzEzb1Nadmebc8KfMnVBUMOHJ_fo_kXmFEZ2vVP70KkO0JfP_imqKzqrFmr2PcwG1egHFtqYNIuwuBlGPaihgt6WzWLyBDvc2R7EMOPgLOsKY1VU-SgHs18jkv59qxYLWyeCnfLo88_cstvYpQygQcHqg05iKghON5vYXtHjiUaHM45dKs2eBoMiAmUKu09tcs6j93HJBOjJSF850VYd05UpW1PSgwPFabdDnlwHm7-xTAHk.9L3Pp4zftTMPPQaXuohixdM7KOBZGOdJMPSt1_RXcTk&dib_tag=se&keywords=k120&qid=1718309899&s=electronics&sprefix=k120%2Celectronics%2C142&sr=1-2&th=1/"> Link </a> |
| Logitech B100 Mouse | Input and selection | $7.99 | <a href="https://www.amazon.com/Logitech-B100-Corded-Mouse-Computers/dp/B003L62T7W/ref=sr_1_3?crid=2GLUL6WJZ0GKO&dib=eyJ2IjoiMSJ9.OKAfwMtMmgjzpEXrJp10_w8xKaMtq7qsCFw-slfV25FJ6ELYelI8G81zHARc8xMbnTCq0tL_OChdFmyNgEhRPUoxERchBVR8gjhwMqhTFISEKzIPDAg4q4-67bUtJ9QuR-JyYdy4QKrLb_eqwybdizcPq1iZbiJ7LZNoIMVa6qXXi_bSBFNF3n90LwKkWHf0m7aNz-YVruwux6_LjHomLs7nRuOJZq9HAm_VWolRxoC5zXDEE4HmjvR3PZX3RyQ3xegWDG9tbdSWfNdPpzPbkSI2vTYmhCHUokWoZ-0Po-g.clnJylxu8WzF8OdB44ZsJ2Hct5aYZCMng0HDx-eXojM&dib_tag=se&keywords=mouse%2Blogitech%2Bwired&qid=1718309962&s=electronics&sprefix=mouse%2Blogitech%2Bwir%2Celectronics%2C137&sr=1-3&th=1/"> Link </a> |


<h2> Starter Project </h2> 


<iframe width="560" height="315" src="https://www.youtube.com/embed/fD4Bt4kgAtw" title="Kallesh A. Starter Project" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<h4>Summary</h4>

My starter project was a retro arcade game. It is a handheld game with includes five classic retro games: Tetris, Snakes, Racing Cars, Space Invaders, and Slots Machine. It is a fun and educational journey as you have to build it yourself. It is a great project once completed to use with family and friends. It is easy to assemble as the instructions are very straight forward. Its main features are six large buttons with rounded acrylic case corners for a comfortable user experience. The game’s working voltage is DC 3.5-5.5V. You can play games with a power bank, 5v power adapter or 3 AAA batteries.

<h4>Challenges </h4>

Some challenges I faced was that I was missing a screw wich caused some difficulty and had to position my other screws in a way to keep it as stable as possible and also my lack of soldering caused the timer at the top to be placed in tilted and since I put too much solder it was extremely hard to remove the solder but it still worked overall. 
