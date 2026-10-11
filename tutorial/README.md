# OSGeo Slovenija - TorchGeo tutorial (1st ed.)

14.10. 2026

Once you have cloned (or loaded) this repository, make sure you are in the root. 
I recommend that you make a new conda environment and isntall all dependencies. 

```bash
conda create -n torchgeo_env python=3.12
conda activate torchgeo_env

cd <root>
pip install .
```

All code needed for tutorial are in the `tutorial` directory. 
Additional download all the data requires from [here](X). And place it inside `tutorial` (both the data and checkpoints folder).

To test if your environment is properly setup, try executing all the intro in `0_intro.ipynb`, 
(we will still go through it together).
You can also check the rest, but we will solve them together.

Solutions are provided in `tutorial/solutions` but I recommend that you rather follow the tutorial. 
Solutions might also change up to the day of tutorial.

This is first time ever I'm doing anything like this, so I'll be very happy to hear any feedback. 
And hopefully it works out fine and you'll enjoy it.