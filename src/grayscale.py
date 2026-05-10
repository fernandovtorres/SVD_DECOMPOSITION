import cv2

def main():
    video_source = cv2.VideoCapture('./data/videoplayback')

    width = int(video_source.get(3))
    height = int(video_source.get(4))

    size = (width, height)

    result = cv2.VideoWriter('./processed_data/grayscale.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             23.976, size, 0)
    

    while True:
        ret, img = video_source.read()

        if not ret:
            break
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result.write(gray)
    
        cv2.imshow("Live", gray)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print("fim")
    cv2.destroyAllWindows()
    video_source.release()


if '__main__' == __name__:
    main()

