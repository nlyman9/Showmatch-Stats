# Download the videos for use in the get_showmatch_stats.py file
# Code adapted from geeksforgeeks.org

# importing the module
from pytube import YouTube
import sys

def get_urls(start, end):
    # Get list of links to download in a range from the playlist
    f = open("video_urls.txt", "r")
    d = f.readlines()
    url_list = []
    for i in d:
        i = i.split(' - ')
        position = int(i[0])
        if position <= end and position >= start:
            substr = i[1][:-1]
            dict = {
                "position": position,
                "link": substr
            }
            url_list.append(dict)
        if position == end:
            break

    return url_list


if __name__ == '__main__':
    # Ensure correct number of arguments. Needs a start and end for the range of videos in the playlist to download
    if len(sys.argv) != 3:
        sys.exit()
    # where to save
    SAVE_PATH = "D:\Homework\Personal Projects\Showmatch Stats\Videos"  # to_do

    answer = input("Preparing to download playlist videos from position " + sys.argv[1] + " to " + sys.argv[2] + ". Continue?(y/n)")
    # Get the url list for the videos that should be downloaded
    if answer == 'y':
        url_list = get_urls(int(sys.argv[1]), int(sys.argv[2]))
    else:
        sys.exit()

    stream = ''
    for dict in url_list:
        try:
            # object creation using YouTube which was imported in the beginning
            yt = YouTube(dict["link"])
            # Get the stream of 720p and 30fps
            ls = yt.streams.filter(fps=30, res="720p", subtype="mp4")
            # Select the first stream
            stream = ls[0]
        except:
            print("Connection Error")  # to handle exception
            
        try:
            # downloading the video
            stream.download(output_path=SAVE_PATH, filename='vid{}'.format(dict["position"]))
        except:
            print("Some Error!")

    print('Task Completed!')



