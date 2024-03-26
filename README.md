 # Santa's Building Detection System

Help Santa detect and segment building footprints in order to make it down every chimney in the world on Christmas!

![image](https://github.com/openSourcerer9000/sbds/assets/61931826/2dd7b4fd-21e9-4343-b94e-c6aa58318b79)

 ## Quickstart
Use `GetBuildingsExample(.ipynb|.py)` to access a new building detection model, fine-tuned from YOLOv8, paired with Meta's Segment Anything to detect and segment building footprints for an arbitrarily large raster. Passing a vector file for the argument `extents` will download imagery for your location to use. Spatial affinement handled by samgeo. The underlying models are built for speed so strap in for the ride!

![image](https://github.com/openSourcerer9000/sbds/assets/61931826/2e106f19-702a-4fd2-8a05-6824b7bdff38)
 - Segment Anything: https://segment-anything.com/
 - Segment Geospatial (samgeo): https://github.com/opengeos/segment-geospatial
 - YOLO: https://docs.ultralytics.com/

Free software: MIT license




This package was created with Cookiecutter_ and the `openSourcerer9000/cookiecutta`_ project template.
.. _`openSourcerer9000/cookiecutta`: https://github.com/openSourcerer9000/cookiecutta
