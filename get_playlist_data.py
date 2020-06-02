import json
import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import requests

#Return a text file containing all of the urls of each video in Johnnyboi_i's 1v1 showmatch playlist


class get_playlist_data:

    def __init__(self):
        self.youtube_client = self.get_youtube_client()
        self.youtube_playlist_id = "PLItVlu8kAxUR9ULwkXOnaBXmtfmdKDjbL"

    # Log into YouTube
    def get_youtube_client(self):
        """Log into youtube, Copied from Youtube Data API"""

        # Disable HTTPS when running locally
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        api_service_name = "youtube"
        api_version = "v3"
        client_secrets_file = "client_secret.json"

        # Get credentials and create an API client
        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
        credentials = flow.run_console()

        # from the Youtube DATA API
        youtube_client = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)
        return youtube_client

    # Grab the Videos from the Playlist
    def get_youtube_playlist_videos(self):
        """Get all the videos from the desired playlist"""

        request = self.youtube_client.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=self.youtube_playlist_id,
            maxResults="50"
        )
        response = request.execute()

        nextpagetoken = response.get('nextPageToken')
        while 'nextPageToken' in response:
            nextpage = self.youtube_client.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=self.youtube_playlist_id,
                maxResults="50",
                pageToken=nextpagetoken
            ).execute()
            response["items"] = response["items"] + nextpage["items"]

            if 'nextPageToken' not in nextpage:
                response.pop('nextPageToken', None)
            else:
                nextpagetoken = nextpage['nextPageToken']

        playlist_videos_info = []
        # Get all videos information in the playlist
        for item in response["items"]:
            # Append video_id to 'https://www.youtube.com/watch?v=' to get the full url
            video_url = item["contentDetails"]["videoId"]
            video_url = "https://www.youtube.com/watch?v=" + video_url
            video_title = item["snippet"]["title"]
            video_position = item["snippet"]["position"] + 1
            video_info = [video_position, video_url, video_title]
            playlist_videos_info.append(video_info)

        return playlist_videos_info


if __name__ == '__main__':
    pl = get_playlist_data()
    print("Found client")
    playlist_videos_info = pl.get_youtube_playlist_videos()
    print("Retrieved playlist info")
    # Have the List of video information, now write to file
    filename = "Showmatches-Playlist.txt"
    playlist_file = open(filename, "w")
    playlist_file.write("Video Information from 1v1 Showmatches Playlist\n")
    for vid in playlist_videos_info:
        playlist_file.write(str(vid[0]) + " , " + vid[1] + " , " + vid[2] + "\n")
    playlist_file.close()
