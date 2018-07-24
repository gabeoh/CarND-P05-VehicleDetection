# CarND-P05-VehicleDetection

CarND-P04-CarND-P05-VehicleDetection implements a pipeline to detect and
track vehicles from images and video streams. 

## File Structure
### Project Requirements
- **[py-src/](py-src/)** - Contains Python source codes that implement the
    pipeline
- **[output_images/](output_images/)** - Contains resulting output images and
    videos for each pipeline step
- **[writeup_report.md](writeup_report.md)** - Project write-up report

### Additional Files
- **[results/](results/)** - Project outputs such as pickle and execution
    log files
- **[test_images/](test_images/)** - Provided test images
- **[test_videos/](test_videos/)** - Provided test videos

### Not Included
- **training_images** - Images to train classifier.  Can be downloaded from
  following links.
  - [Vehicle Images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
  - [Non-Vehicle Images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
  - [Udacity Labeled Training Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)

## Getting Started
### [Download ZIP](https://github.com/gabeoh/CarND-P05-VehicleDetection/archive/master.zip) or Git Clone
```
git clone https://github.com/gabeoh/CarND-P05-VehicleDetection.git
```

### Setup Environment

You can set up the environment following
[CarND-Term1-Starter-Kit - Miniconda](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md).
This will install following packages required to run this application.

- Miniconda
- Python

### Download Simulator
- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)


### Usage

#### Run Vehicle Detection on Test Images
```
$ cd py-src

$ python p05_vehicle_detection_main.py --image
or
$ python p05_vehicle_detection_main.py -i
```

You can also run only specific steps.  For example, run only step 3
perspective transform and step 4 lane line identification.
```
$ python p05_vehicle_detection_main.py -i -s 3 4
```

More information on running option can be found by running:
```
$ python p05_vehicle_detection_main.py -h
```

#### Run Lane Detection on Test Videos
```
$ cd py-src

$ python p05_vehicle_detection_main.py --video
or
$ python p05_vehicle_detection_main.py -v
```

You can also run on only specific video files.  For example, run the pipeline
only on project video.
```
$ python p05_vehicle_detection_main.py -v -f project_video.mp4
```

More information on running option can be found by running:
```
$ python p05_vehicle_detection_main.py -h
```

## License
Licensed under [MIT](LICENSE) License.

