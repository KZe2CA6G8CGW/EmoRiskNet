if __name__ == "__main__":
    # 模拟输入（e.g. 4 个模态）
    input_channels = {
        'imaging': 1,
        'lab': 1,
        'demographic': 1,
        'history': 1
    }

    model = MedForm(input_channels_dict=input_channels)

    inputs = {
        'imaging': torch.randn(8, 1, 64, 64),
        'lab': torch.randn(8, 1, 64, 64),
        'demographic': torch.randn(8, 1, 64, 64),
        'history': torch.randn(8, 1, 64, 64),
    }

    # masks: 模态是否缺失（False = 缺失）
    masks = {
        'imaging': True,
        'lab': True,
        'demographic': False,
        'history': True,
    }

    output, embeddings = model(inputs, masks)
    print("Logits:", output.shape)
    print("Hierarchy embeddings:", len(embeddings))
