declare -a names=("ResNet18" "ResNet50" "ResNeXt")
# declare -a names=("ResNet18" "ResNet50" "ResNeXt" "ConvNeXt" "ViT" "DinoV2")
declare -a layers=("layer3" "layer4")

for name in "${names[@]}"; do
    for layer in "${layers[@]}"; do
        echo "Finetuning $name..."
        python ./main.py --name "$name" -f --epochs 30 --layer "$layer$
    done
done