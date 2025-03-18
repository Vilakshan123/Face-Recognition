def recognize_and_visualize_batch(folder_paths, actual_labels, num_samples=20):
    """
    Recognizes faces in the first `num_samples` images found inside given folders and displays them.
    """
    plt.figure(figsize=(15, 12))  # Adjust figure size for visibility
    sample_count = 0  # Track number of processed images

    for i, folder_path in enumerate(folder_paths):
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Error: Folder {folder_path} not found.")
            continue

        # Get the first image file in the directory
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"Error: No image files found in {folder_path}.")
            continue

        image_path = os.path.join(folder_path, image_files[0])  # Pick the first image

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load {image_path}.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face, coordinates = detect_face(img_rgb)  # Ensure detect_face() returns (face, (x, y, w, h))

        if face is not None:
            embedding = embedder.embeddings(np.expand_dims(face, axis=0))
            proba = clf.predict_proba(embedding)
            confidence = np.max(proba)
            pred = clf.predict(embedding)[0]

            # Bounding box adjustments for better visibility
            x, y, w, h = coordinates
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display actual & predicted labels
            label = f"Pred: {pred} ({confidence:.2f})\nActual: {actual_labels[i]}"
            text_x, text_y = x, max(y - 10, 20)  # Avoid placing text outside image
            cv2.putText(img_rgb, f"{pred} ({confidence:.2f})", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            label = f"No face detected\nActual: {actual_labels[i]}"

        # Plot the image with labels
        plt.subplot(4, 5, sample_count + 1)  # Create a 4x5 grid
        plt.imshow(img_rgb)
        plt.title(label, fontsize=10)
        plt.axis("off")

        sample_count += 1
        if sample_count >= num_samples:  # Stop after 20 images
            break

    plt.tight_layout()
    plt.show()

# Example usage (Replace these paths & labels with your actual dataset)
folder_paths = [
   "/content/predata/predata/000638",
    "/content/predata/predata/000651",
    "/content/predata/predata/000654",
    "/content/predata/predata/000659",
    "/content/predata/predata/000673",
    "/content/predata/predata/000697",
    "/content/predata/predata/000732",
   "/content/predata/predata/000693",
   "/content/predata/predata/000674",
   "/content/predata/predata/000690",
   "/content/predata/predata/000688"

]

actual_labels = [
    "000638", "000651", "000660", "000659", "000673", "000697", "000732","000693","000674","000690","000688"
]

# Call function for visualization
recognize_and_visualize_batch(folder_paths, actual_labels)