.PHONY: all requirements clone dataset checkpoints clean

all:
	echo "Setting up the project..."
	$(MAKE) requirements
	$(MAKE) clone
	$(MAKE) dataset
	$(MAKE) checkpoints

requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

clone:
	mkdir -p models_src
	git clone https://github.com/CompVis/taming-transformers.git models_src/VQGAN
	git clone https://github.com/kakaobrain/rq-vae-transformer.git models_src/RQTransformer
	git clone https://github.com/FoundationVision/LlamaGen.git models_src/LlamaGen
	git clone https://github.com/FoundationVision/VAR.git models_src/VAR

dataset:
	mkdir -p dataset
	# download the imagenet validation split, name it imagenet-val 
	wget https://www.kaggle.com/api/v1/datasets/download/titericz/imagenet1k-val --no-check-certificate -O dataset/imagenet-val.zip
	unzip dataset/imagenet-val.zip -d dataset/imagenet-val
	rm dataset/imagenet-val.zip

	# download the metadata for imagenet ILSVRC2012_devkit_t12.tar.gz
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate -O dataset/ILSVRC2012_devkit_t12.tar.gz

	# download the metadata for imagenet ILSVRC20212_img_val.tar
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate -O dataset/ILSVRC2012_img_val.tar

checkpoints:
	# get checkpoints
	mkdir -p checkpoints/LlamaGen
	mkdir -p checkpoints/RQTransformer
	mkdir -p checkpoints/VQGAN
	mkdir -p checkpoints/VAR

	# download LlamaGen checkpoints
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_B_256.pt -O checkpoints/LlamaGen/c2i_B_256.pt
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_B_384.pt -O checkpoints/LlamaGen/c2i_B_384.pt
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_L_256.pt -O checkpoints/LlamaGen/c2i_L_256.pt
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_L_384.pt -O checkpoints/LlamaGen/c2i_L_384.pt
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XL_384.pt -O checkpoints/LlamaGen/c2i_XL_384.pt
	wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_XXL_384.pt -O checkpoints/LlamaGen/c2i_XXL_384.pt

	# download RQTransformer checkpoints
	wget https://twg.kakaocdn.net/brainrepo/models/RQVAE/7518a004fe39120fcffbba76005dc6c3/imagenet_480M.tar.gz --no-check-certificate -O checkpoints/RQTransformer/imagenet_480M.tar.gz
	tar -xvzf checkpoints/RQTransformer/imagenet_480M.tar.gz -C checkpoints/RQTransformer
	rm checkpoints/RQTransformer/imagenet_480M.tar.gz

	wget https://twg.kakaocdn.net/brainrepo/models/RQVAE/dcd39292319104da5577dec3956bfdcc/imagenet_821M.tar.gz --no-check-certificate -O checkpoints/RQTransformer/imagenet_821M.tar.gz
	tar -xvzf checkpoints/RQTransformer/imagenet_821M.tar.gz -C checkpoints/RQTransformer
	rm checkpoints/RQTransformer/imagenet_821M.tar.gz

	wget https://twg.kakaocdn.net/brainrepo/models/RQVAE/f5cf4e5f3f0b5088d52cbb5e85c1077f/imagenet_1.4B.tar.gz --no-check-certificate -O checkpoints/RQTransformer/imagenet_1.4B.tar.gz
	tar -xvzf checkpoints/RQTransformer/imagenet_1.4B.tar.gz -C checkpoints/RQTransformer
	rm checkpoints/RQTransformer/imagenet_1.4B.tar.gz

	# download VAR checkpoints
	wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth -O checkpoints/VAR/vae_ch160v4096z32.pth
	wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth -O checkpoints/VAR/var_d16.pth
	wget https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth -O checkpoints/VAR/var_d20.pth
	wget https://huggingface.co/FoundationVision/var/resolve/main/var_d24.pth -O checkpoints/VAR/var_d24.pth
	wget https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth -O checkpoints/VAR/var_d30.pth

	# download VQGAN checkpoints
	wget https://app.koofr.net/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e?path=%2F2021-04-03T19-39-50_cin_transformer%2Fconfigs%2F2021-04-03T19-39-50-project.yaml -O checkpoints/VQGAN/configs.yaml
	wget https://app.koofr.net/content/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e/files/get/last.ckpt?path=%2F2021-04-03T19-39-50_cin_transformer%2Fcheckpoints%2Flast.ckpt -O checkpoints/VQGAN/last.ckpt

clean:
	rm -rf models_src
	rm -rf dataset
	rm -rf checkpoints
