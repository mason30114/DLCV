python -c "import clip; clip.load('ViT-B/32')"

python -c "import timm; timm.create_model('vit_gigantic_patch14_clip_224', pretrained=True)"

wget -O p2_lora.pth https://www.dropbox.com/scl/fi/zduifaexza8p1kdrb395n/p2_lora.pth?rlkey=8ik2wiso7yp3qxpvgpy5n4q9u&dl=1
