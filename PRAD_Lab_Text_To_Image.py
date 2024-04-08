#!/usr/bin/env python
# coding: utf-8

# In[6]:


# exclude warning
import warnings
warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame instead")

# import library 
from IPython.display import HTML

# embed video
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/SVcsDDABEkM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# In[7]:


# Install required libraries
get_ipython().system('pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy')


# In[9]:


from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from torch import autocast


# In[10]:


# Set up GPU environment
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)


# In[11]:


# Set up pipeline with Stable Diffusion model and scheduler
model_id = "stabilityai/stable-diffusion-2"


# In[12]:


# Load scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")


# In[13]:


# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(torch_device)


# In[16]:


# Ensure torch is using float32
torch.set_default_dtype(torch.float32)


# In[17]:


# Set up pipeline with Stable Diffusion model and scheduler
model_id = "stabilityai/stable-diffusion-2"


# In[18]:


# Load scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")


# In[19]:


# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32)
pipe = pipe.to(torch_device)


# In[20]:


# Prompt
prompt = 'A steampunk woman reading a book with her cat at the bar'

# Inference
image = pipe(prompt, height=768, width=768).images[0]
image


# In[21]:


# Prompt
prompt = 'A businessman at a pub'

# Inference
image = pipe(prompt, height=768, width=768).images[0]
image


# In[22]:


# Prompt
prompt = 'Peaky Blinders'

# Inference
image = pipe(prompt, height=768, width=768).images[0]
image


# In[ ]:




