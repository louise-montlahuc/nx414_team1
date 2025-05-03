declare -a names=("DinoV2" "ViT" "ConvNeXt" "ResNeXt")
# declare -a names=("ResNet18" "ResNet50" "ResNeXt" "ConvNeXt" "ViT" "DinoV2")
declare -a probing=("linear" "ridge")
# declare -a probing=("linear" "ridge" "mlp")

for name in "${names[@]}"; do
    for probe in "${probing[@]}"; do
        echo "Running script for $name with probing $probe"
        python ./main.py --name "$name" --probing "$probe" --saved
    done
done