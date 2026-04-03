**Assignment-5** 

Deadline**: 03/04/2026 11:59 PM**  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**Submission Guidelines:**

1. Use Python and PyTorch for performing experiments.  
2. **Report file naming convention:  Rollnumber\_Name\_Ass5.pdf. For example: B22CS043\_Firstname\_Surname\_Ass5.pdf**  
3. **On Google Classroom, upload the link to the GitHub branch for Assignment \- 5, the WandB and HuggingFace links, and a single report in .pdf format only.**  
4. Follow all the GitHub guidelines mentioned below.  
5. **For this Assignment, you are not allowed to use Colab or .ipynb files.**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**Github Guidelines:**

* **Create a branch name Assignment 5\.**  
* **Create a README.md file in which you have to write how to install the libraries you have used, and how to run the training and testing codes.**  
* **Push requirements.txt, .py and report (.pdf) files.**  
* **Push weights of the best model for Q1, and all weights of Q2.**  
* **Add train-val tables and graphs for both Q1 and Q2, and add samples of qualitative image results of Q2.**  
* **The README should also contain the tables and results.**  
* **Add the WandB, HuggingFace link in GitHub README and Report.**  
* **Update your GitHub page for this assignment.**  
* **GitHub should be properly structured so that it clearly explains how to install the libraries and the commands to run the training and testing codes.**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**NOTE: The questions below must be done using a Docker container.**

**Q1. Take a ViT-S (small) model that is already pre-trained on ImageNet. Follow the steps below for the CIFAR-100 dataset:**

1. **Finetune ViT classification head for 100 classes without LoRA.**  
2. **Finetune ViT by applying LoRA using PEFT for various LoRA hyperparameters mentioned below, along with keeping the classification head trainable for 100 classes.**  
   1. **Inject LoRA into attention weights of Q, K, V**  
   2. **Perform experiments for Ranks: 2,4,8**  
   3. **Alpha: 2,4,8**  
   4. **Dropout: 0.1**

	**Perform experiments for all combinations of a and b to complete steps 3 and 4 based on this.**

3. **Show results in the report and also upload on WandB, including Train-val Loss and Accuracy Tables (mentioned below) and Graphs, Class-wise Test Accuracy Histogram, and Gradient update graphs on LoRA weights during Training.**  
     
     
     
     
   **Experiment No. (based on combinations in step 2): \_\_\_\_\_\_, Rank: \_\_\_\_\_\_\_\_, Alpha: \_\_\_\_\_\_\_\_\_\_\_**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| :---- | :---- | :---- | :---- | :---- |
| **1** |  |  |  |  |
| **2** |  |  |  |  |
| **3** |  |  |  |  |
| **4** |  |  |  |  |
| **5** |  |  |  |  |
| **6** |  |  |  |  |
| **7** |  |  |  |  |
| **8** |  |  |  |  |
| **9** |  |  |  |  |
| **10** |  |  |  |  |

   

4. **Create a table for Testing with the following columns for your results (Also upload all the results in the Readme file of GitHub):**

| LoRA layers (with/without) | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters used |
| :---- | :---- | :---- | :---- | :---- | :---- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

5. **Use Optuna to find the best hyperparameter configuration for LoRA-hyperameters only.**  
6. **Push the best model weights to GitHub and HuggingFace.**   
7. **\[Optional\] Perform the experiment for the best hyperparameter configuration observed in step 2 using LoRA, keeping the original model partially frozen and partially trainable (along with a trainable classification head) and applying LoRA to the partially frozen part. Report your observations (Table in steps 3 and 4\) and analysis based on this.**

**Q2. Adversarial Attacks using IBM Adversarial Robust Toolkit (ART)**

**(i) Task: FastGradientMethod (FGSM) Attack: From Scratch vs IBM ART**  
   **On the CIFAR-10 dataset using a (non-pretrained) ResNet18 model, follow the steps below:**

1. **Train on clean CIFAR-10 samples from scratch, achieving ≥ 72% test classification accuracy.**  
2. **Implement the FGSM attack from scratch (without ART).**  
3. **Then implement the FGSM attack using IBM ART.**  
4. **Show visual comparison: Original vs adversarial images with and without IBM ART.**  
5.  **Report and Compare:**  
   * **Accuracy (clean vs adversarial without IBM ART vs adversarial with IBM ART)**  
   * **Perturbation strength vs performance drop**  
6. **Show and analyze the impact of the attack with and without IBM ART.**

**(ii) Task: Adversarial Detection Model**  
     **Design and implement deep learning-based adversarial detectors using the ResNet-34 model using CIFAR-10 dataset:**  
         	**(a)**   
**Input:**

* **Mix of clean \+ adversarial images created using PGD (Projected Gradient Descent) attack through IBM ART.**

**Output:**

* **Binary classification (clean vs adversarial)**

  **(b)** 

**Input:**

* **Mix of clean \+ adversarial images created using BIM (Basic Iterative Method) attack through IBM ART.**

**Output:**

* **Binary classification (clean vs adversarial)**

	  
	**Report:**

* **Detection accuracy in each case must be ≥ 70%**  
* **Compare detection performance across different attacks: PGD and BIM.**

**On the WandB show 10 samples of its clean and adversarial images created using FGSM (with and without IBM ART), PGD, and BIM attacks.**

	
