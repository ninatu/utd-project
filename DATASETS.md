# Datasets

We considered **12 video datasets** in total for our experiments.

## 🔗 Downloading Metadata

To simplify processing, we aligned the metadata across all datasets.  
You can download it from the following link and place it in the `./metadata` directory:  
🔗 [**GDrive link.**](https://drive.google.com/drive/folders/1Ks597ykP3of-lsIcIsrL0SHHt7Ke64Wt?usp=sharing)

> **Note:** You may need to adjust the `full_path` entries depending on where and how you store your videos.


## 🧪 Test Splits Used

Following prior work, we use the following splits for evaluation:

- **`test` splits**: `msrvtt`, `didemo`, `lsmdc`, `S-MiT`

- **`val` splits**: `kinetics_400`, `kinetics_600`, `kinetics_700`, `MiT`, `activity_net`, `youcook`, `ssv2`

---

## 🎞️ Downloading Videos

- **MSRVTT**, **LSMDC**, and **DiDeMo**: Follow the [CLIP4CLIP guidelines](https://github.com/ArrowLuo/CLIP4Clip) to download videos of these datasets. We store videos in the following folders:

  - `data/msrvtt/videos/all/`

  - `data/lsmdc/avi/` (with subfolders)

  - `data/didemo/videos/`

- **Kinetics 400, 600, 700**: Follow the [Kinetics dataset instructions](https://github.com/cvdfoundation/kinetics-dataset) to download. We store videos in the following folders:

  - `data/kinetics_400/{train,val}_data_resized/` (with class subfolders)

  - `data/kinetics_600/videos/{train,val,test}/` (with class subfolders)

  - `data/kinetics_700/kinetics-dataset/k700-2020/{train,val,test}/` (with class subfolders)

- **MiT** and **S-MiT**: Follow the [Moments in Time website](http://moments.csail.mit.edu/) for download instructions. We store videos in the following folders:

  - `data/MiT/Moments_in_Time_Raw/{training,validation}/` (with class subfolders)

  - `data/S-MiT/videos/` (with class subfolders)

- **ActivityNet**: Follow the official [ActivityNet download instructions](http://activity-net.org/). Access the full dataset via Google Drive or Baidu after submitting a request. We used only the downsampled version `Anet_videos_15fps_short256`:

  - `data/activity_net/Anet_videos_15fps_short256/`

- **YouCook2**: Follow the [YouCook2 guidelines](http://youcook2.eecs.umich.edu/) to download. We store videos in the following folders:

  - `data/youcook/{training,validation,testing}/`

- **UCF101**:  Download from the [official UCF101 website](https://www.crcv.ucf.edu/data/UCF101.php). Store the videos in:

  - `data/UCF101/original/videos/`

- **Something-Something v2**: Follow the [official instructions](https://www.qualcomm.com/developer/software/something-something-v-2-dataset). We converted all videos to `.mp4` and stored them in:

  - `data/SomethingSomething_v2/ssv2/train_data_mp4_correct/`

  - `data/SomethingSomething_v2/ssv2/val_data_mp4_correct/`