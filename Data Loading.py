def load_data(dataset_path, num_classes=200, samples_per_class=10):
    embeddings = []
    labels = []

    class_folders = sorted(os.listdir(dataset_path))[:num_classes]

    for folder_name in tqdm(class_folders):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)[:samples_per_class]

            for image_file in images:
                image_path = os.path.join(folder_path, image_file)
                img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                # Original image
                face, _ = detect_face(img)
                if face is not None:
                    embedding = embedder.embeddings(np.expand_dims(face, axis=0))
                    embeddings.append(embedding[0])
                    labels.append(folder_name)

                # Augmented images
                for augmented in datagen.flow(np.expand_dims(img, axis=0), batch_size=1):
                    aug_face, _ = detect_face(augmented[0].astype('uint8'))
                    if aug_face is not None:
                        embedding = embedder.embeddings(np.expand_dims(aug_face, axis=0))
                        embeddings.append(embedding[0])
                        labels.append(folder_name)
                    break  # Generate 1 augmented version per image

    return np.array(embeddings), np.array(labels)

# Load dataset with augmentation
X, y = load_data("/content/predata/predata")