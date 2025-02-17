try:
    import mobileclip_stuff
    mobileclip_ok=True
except ImportError:
    mobileclip_ok=False

class LLMMobileCLIP:
    def __init__(self):
       assert mobileclip_ok, "Need to set up mobileclip stuff"
       self.state=mobileclip_stuff.mobileclip_init()

    def get_batch(self):
        return 2048
    
    def get_max_size(self):
        return 512,512

    def generate_attributes(self, attrs, jpegs):
        return mobileclip_stuff.mobileclip_generate_attributes(self.state, attrs, jpegs)