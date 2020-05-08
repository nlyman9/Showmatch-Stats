from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import sys
import uuid
from multiprocessing import Process


class getShowMatchStats:

    def __init__(self):
        self.initialendgametime = 9000  # Take the first screen capture after 5 minutes
        self.endgametime = -1  # How long is currently left on the timer
        self.initial_frame = True  # Flag for getting the initial timer capture
        self.timer_zero = False  # Flag for if the timer equals zero
        self.winner_chosen = False  # Flag for if winner has been displayed

        # self.frame_stats_score_top = [305, 335, 745, 805]  # Dimensions for capturing the score on top
        # self.frame_stats_score_bot = [410, 440, 745, 805]  # Dimensions for capturing the score on bottom
        self.verify_previous_timer_val = -1  # Store the previous timer value so current timer value is accurate

        # GAME 1 STATS POSITIONING
        self.frame_top_name = [300, 325, 513, 715]  # Dimensions for capturing the top name on the scoreboard
        self.frame_bot_name = [405, 430, 513, 715]  # Dimensions for capturing the bottom name on the scoreboard
        self.frame_stats_goals_top = [305, 335, 820, 855]
        self.frame_stats_goals_bot = [410, 440, 820, 855]
        self.frame_stats_assists_top = [305, 335, 885, 925]
        self.frame_stats_assists_bot = [410, 440, 885, 925]
        self.frame_stats_saves_top = [305, 335, 960, 995]
        self.frame_stats_saves_bot = [410, 440, 960, 995]
        self.frame_stats_shots_top = [305, 335, 1015, 1055]
        self.frame_stats_shots_bot = [410, 440, 1015, 1055]

        # GAME 2 STATS POSITIONING
        self.frame_top_name2 = [285, 310, 513, 715]  # Dimensions for capturing the top name on the scoreboard
        self.frame_bot_name2 = [390, 415, 513, 715]  # Dimensions for capturing the bottom name on the scoreboard
        self.frame_stats_goals_top2 = [290, 320, 820, 855]
        self.frame_stats_goals_bot2 = [395, 425, 820, 855]
        self.frame_stats_assists_top2 = [290, 320, 885, 925]
        self.frame_stats_assists_bot2 = [395, 425, 885, 925]
        self.frame_stats_saves_top2 = [290, 320, 960, 995]
        self.frame_stats_saves_bot2 = [395, 425, 960, 995]
        self.frame_stats_shots_top2 = [290, 320, 1015, 1055]
        self.frame_stats_shots_bot2 = [395, 425, 1015, 1055]

    # Do Optical Character Recognition on img
    def ocr(self, img, config):
        lang = False  # Flag for if using a trained language or not
        langtimer = False  # Flag for the timer language

        if config == "timer":
            langtimer = True
            custom_config = r'-c tessedit_char_whitelist=0123456789: --psm 7'
        elif config == "winner":
            custom_config = r'-c tessedit_char_blacklist=., --psm 8'
        elif config == "single":
            # print("Using single digit")
            lang = True
            custom_config = r' --psm 10 outputbase digits'
        else:
            custom_config = r'--psm 7'
        pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract.exe'
        newImage = cv2.bitwise_not(img)
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filename = "{}.jpg".format(str(uuid.uuid4()))
        cv2.imwrite(filename, gray)
        if lang:
            text = pytesseract.image_to_string(Image.open(filename), lang='rl', config=custom_config)
        elif langtimer:
            text = pytesseract.image_to_string(Image.open(filename), lang='langtimer', config=custom_config)
        else:
            text = pytesseract.image_to_string(Image.open(filename), config=custom_config)
        os.remove(filename)
        print("Text", text)
        # show the output images
        # cv2.imshow("Image", img)
        cv2.imshow("Output", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return text

    def editvideo(self, vid, bar, era):
        cap = cv2.VideoCapture(vid)
        count = 0
        # Play the video
        while cap.isOpened():
            ret, frame = cap.read()
            # If the video is over, end
            if not ret:
                break
            # grayscale it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.initial_frame and count == self.initialendgametime:
                # Grab the frame after one minute and see what the timer says
                self.get_timer_value(frame, bar)
                self.initial_frame = False
                count = 0
            elif not self.initial_frame and count == self.endgametime and not self.winner_chosen:
                # Wait however many seconds are left in game, then take screenshot of timer
                if self.timer_zero is False:
                    self.get_timer_value(frame, bar)
                # Check if the timer is 0
                if self.endgametime == 0 or self.timer_zero is True:
                    self.timer_zero = True
                    # Check timer every second to get when the ball actually hits the ground ending the game
                    winner = self.wait_ball_hit_ground(frame, bar, era)
                    print("Winner is currently:", winner)
                    if winner == "winner":
                        print("Winner winner chicken dinner")
                        self.winner_chosen = True
                        # The game has officially ended
                count = 0
            elif self.winner_chosen:
                # Wait 13 seconds for stats screen to show
                if era == 1:
                    self.endgametime = 150
                else:
                    self.endgametime = 390

                if count == self.endgametime:
                    # Get information from the stats screen
                    self.capture_stats_screen(frame, bar)
                    cv2.waitKey(0)
                    self.initial_frame = True
                    self.timer_zero = False
                    self.winner_chosen = False
                    count = 0
            count += 1
            # print("Count", count, "endgametime", self.endgametime)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Crop the frame so that it just includes the timer using 720p res
    def crop_image_timer(self, frame, bar):
        if bar == "bar":
            cv2.imwrite("timer.jpg", frame)
            crop_img = frame[13:42, 590:695]
        else:
            crop_img = frame[23:62, 590:695]
        return crop_img

    # Get the number of frames until the end of the rocket league game at 30fps
    def calc_new_end_game(self, time):
        digitlist = []
        for digit in range(len(time)):
            if time[digit] == ':':
                print("Digit was :, not appending")
            else:
                digitlist.append(int(time[digit]))
        if len(digitlist) == 3:
            print("digitlist len: ", 3)
            print("digitlist 0: ", digitlist[0])
            print("digitlist 1: ", digitlist[1])
            print("digitlist 2: ", digitlist[2])
            end_game = ((digitlist[0] * 60) + (digitlist[1] * 10) + digitlist[2]) * 30
            if end_game == 0:
                return end_game
            elif self.verify_previous_timer_val == -1:
                # Set the previous timer value and set endgame = 30 to get the next second
                self.verify_previous_timer_val = end_game
                end_game = 30
                print("Setting previous val =", self.verify_previous_timer_val, "and taking a screenshot in next second")
            else:
                # If this == 1 or 0, the timer's time is verified
                if (self.verify_previous_timer_val - end_game) == 30 or (self.verify_previous_timer_val - end_game) == 0:
                    # Verified
                    print("Subtraction equals 0 or 1 so timer is verified")
                    self.verify_previous_timer_val = -1
                else:
                    # Timer is not verified, take another timer value at next second
                    print("Some number was messed up. Taking a screenshot in next second and setting previous =", self.verify_previous_timer_val, "to", end_game)
                    self.verify_previous_timer_val = end_game
                    end_game = 30
        else:
            print("Digitlist is not 3, taking new screenshot after a second")
            end_game = 30  # This makes the timer wait one second before selecting again

        # Store the previous timer value into a global variable
        # Have -1 as the need to set value of the global variable and anything else means you should compare the current val to the previous val
        # To make sure it's valid make sure the previous value - the current value = 1 or 0.

        print("Next timer check is in", (end_game / 30), "second(s)")
        return end_game

    def get_timer_value(self, frame, bar):
        cframe = self.crop_image_timer(frame, bar)
        text = self.ocr(cframe, "timer")
        self.endgametime = self.calc_new_end_game(text)
        print("endgametime = ", self.endgametime)
        # We have already gotten the frame after 5 minutes so now get the frame once the game ends

    def wait_ball_hit_ground(self, frame, bar, era):
        cframe = self.crop_image_winner(frame, bar, era)
        text = self.ocr(cframe, "winner")
        self.endgametime = 30
        return text.lower()

    def crop_image_stats(self, frame):
        crop_img = frame[20:65, 585:700]
        return crop_img

    def crop_image_winner(self, frame, bar, era):
        if era == 1:
            crop_img = frame[275:325, 540:740]
        elif bar == "bar":
            crop_img = frame[280:325, 555:725]  # Was 555:735 for x
        else:
            crop_img = frame[290:338, 555:725]
        return crop_img

    def scaleup(self, frame, scale):
        scale_percent = scale*100
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(frame, dsize)
        return output

    # Wait from the time the text "winner" is displayed to the time the stats screen is displayed
    # Convert the stats screen to strings
    def capture_stats_screen(self, frame, orientation):
        if orientation == "bar":
            # Get name for top
            frame_topname = frame[self.frame_top_name2[0]:self.frame_top_name2[1],
                            self.frame_top_name2[2]:self.frame_top_name2[3]]
            frame_topname = self.scaleup(frame_topname, 3)
            # cv2.imwrite("topname.jpg", frame_topname)
            topname = self.ocr(frame_topname, "name")
            print("Top name is:", topname)
            # Get goals for top
            frame_topgoals = frame[self.frame_stats_goals_top2[0]:self.frame_stats_goals_top2[1],
                             self.frame_stats_goals_top2[2]:self.frame_stats_goals_top2[3]]
            frame_topgoals = self.scaleup(frame_topgoals, 3)
            topgoals = self.ocr(frame_topgoals, "single")
            print("Top goals is:", topgoals)
            # Get assists for top
            frame_topassists = frame[self.frame_stats_assists_top2[0]:self.frame_stats_assists_top2[1],
                               self.frame_stats_assists_top2[2]:self.frame_stats_assists_top2[3]]
            frame_topassists = self.scaleup(frame_topassists, 3)
            topassists = self.ocr(frame_topassists, "single")
            print("Top assists is:", topassists)
            # Get saves for top
            frame_topsaves = frame[self.frame_stats_saves_top2[0]:self.frame_stats_saves_top2[1],
                             self.frame_stats_saves_top2[2]:self.frame_stats_saves_top2[3]]
            frame_topsaves = self.scaleup(frame_topsaves, 3)
            topsaves = self.ocr(frame_topsaves, "single")
            print("Top saves is:", topsaves)
            # Get shots for top
            frame_topshots = frame[self.frame_stats_shots_top2[0]:self.frame_stats_shots_top2[1],
                             self.frame_stats_shots_top2[2]:self.frame_stats_shots_top2[3]]
            frame_topshots = self.scaleup(frame_topshots, 3)
            topshots = self.ocr(frame_topshots, "single")
            print("Top shots is:", topshots)
            # Get name for bot
            frame_botname = frame[self.frame_bot_name2[0]:self.frame_bot_name2[1],
                            self.frame_bot_name2[2]:self.frame_bot_name2[3]]
            frame_botname = self.scaleup(frame_botname, 3)
            botname = self.ocr(frame_botname, "name")
            print("Bot name is:", botname)
            # Get goals for bot
            frame_botgoals = frame[self.frame_stats_goals_bot2[0]:self.frame_stats_goals_bot2[1],
                             self.frame_stats_goals_bot2[2]:self.frame_stats_goals_bot2[3]]
            frame_botgoals = self.scaleup(frame_botgoals, 3)
            botgoals = self.ocr(frame_botgoals, "single")
            print("Bot goals is:", botgoals)
            # Get assists for bot
            frame_botassists = frame[self.frame_stats_assists_bot2[0]:self.frame_stats_assists_bot2[1],
                               self.frame_stats_assists_bot2[2]:self.frame_stats_assists_bot2[3]]
            frame_botassists = self.scaleup(frame_botassists, 3)
            botassists = self.ocr(frame_botassists, "single")
            print("Bot assists is:", botassists)
            # Get saves for bot
            frame_botsaves = frame[self.frame_stats_saves_bot2[0]:self.frame_stats_saves_bot2[1],
                             self.frame_stats_saves_bot2[2]:self.frame_stats_saves_bot2[3]]
            frame_botsaves = self.scaleup(frame_botsaves, 3)
            botsaves = self.ocr(frame_botsaves, "single")
            print("Bot saves is:", botsaves)
            # Get shots for bot
            frame_botshots = frame[self.frame_stats_shots_bot2[0]:self.frame_stats_shots_bot2[1],
                             self.frame_stats_shots_bot2[2]:self.frame_stats_shots_bot2[3]]
            frame_botshots = self.scaleup(frame_botshots, 3)
            botshots = self.ocr(frame_botshots, "single")
            print("Bot shots is:", botshots)
        else:
            # Get name for top
            frame_topname = frame[self.frame_top_name[0]:self.frame_top_name[1],
                            self.frame_top_name[2]:self.frame_top_name[3]]
            frame_topname = self.scaleup(frame_topname, 3)
            # cv2.imwrite("topname.jpg", frame_topname)
            topname = self.ocr(frame_topname, "name")
            print("Top name is:", topname)
            # Get goals for top
            frame_topgoals = frame[self.frame_stats_goals_top[0]:self.frame_stats_goals_top[1],
                             self.frame_stats_goals_top[2]:self.frame_stats_goals_top[3]]
            frame_topgoals = self.scaleup(frame_topgoals, 3)
            topgoals = self.ocr(frame_topgoals, "single")
            print("Top goals is:", topgoals)
            # Get assists for top
            frame_topassists = frame[self.frame_stats_assists_top[0]:self.frame_stats_assists_top[1],
                               self.frame_stats_assists_top[2]:self.frame_stats_assists_top[3]]
            frame_topassists = self.scaleup(frame_topassists, 3)
            topassists = self.ocr(frame_topassists, "single")
            print("Top assists is:", topassists)
            # Get saves for top
            frame_topsaves = frame[self.frame_stats_saves_top[0]:self.frame_stats_saves_top[1],
                             self.frame_stats_saves_top[2]:self.frame_stats_saves_top[3]]
            frame_topsaves = self.scaleup(frame_topsaves, 3)
            topsaves = self.ocr(frame_topsaves, "single")
            print("Top saves is:", topsaves)
            # Get shots for top
            frame_topshots = frame[self.frame_stats_shots_top[0]:self.frame_stats_shots_top[1],
                             self.frame_stats_shots_top[2]:self.frame_stats_shots_top[3]]
            frame_topshots = self.scaleup(frame_topshots, 3)
            topshots = self.ocr(frame_topshots, "single")
            print("Top shots is:", topshots)
            # Get name for bot
            frame_botname = frame[self.frame_bot_name[0]:self.frame_bot_name[1],
                            self.frame_bot_name[2]:self.frame_bot_name[3]]
            frame_botname = self.scaleup(frame_botname, 3)
            botname = self.ocr(frame_botname, "name")
            print("Bot name is:", botname)
            # Get goals for bot
            frame_botgoals = frame[self.frame_stats_goals_bot[0]:self.frame_stats_goals_bot[1],
                             self.frame_stats_goals_bot[2]:self.frame_stats_goals_bot[3]]
            frame_botgoals = self.scaleup(frame_botgoals, 3)
            botgoals = self.ocr(frame_botgoals, "single")
            print("Bot goals is:", botgoals)
            # Get assists for bot
            frame_botassists = frame[self.frame_stats_assists_bot[0]:self.frame_stats_assists_bot[1],
                               self.frame_stats_assists_bot[2]:self.frame_stats_assists_bot[3]]
            frame_botassists = self.scaleup(frame_botassists, 3)
            botassists = self.ocr(frame_botassists, "single")
            print("Bot assists is:", botassists)
            # Get saves for bot
            frame_botsaves = frame[self.frame_stats_saves_bot[0]:self.frame_stats_saves_bot[1],
                             self.frame_stats_saves_bot[2]:self.frame_stats_saves_bot[3]]
            frame_botsaves = self.scaleup(frame_botsaves, 3)
            botsaves = self.ocr(frame_botsaves, "single")
            print("Bot saves is:", botsaves)
            # Get shots for bot
            frame_botshots = frame[self.frame_stats_shots_bot[0]:self.frame_stats_shots_bot[1],
                             self.frame_stats_shots_bot[2]:self.frame_stats_shots_bot[3]]
            frame_botshots = self.scaleup(frame_botshots, 3)
            botshots = self.ocr(frame_botshots, "single")
            print("Bot shots is:", botshots)

    # Helper function to do an OCR on a specific part of a frame of a video that happens after seconds_into_video seconds
    def get_ocr(self, vid, seconds_into_video, topy, topyplush, topx, topxplusw):
        cap = cv2.VideoCapture(vid)
        count = 0
        numFrames = seconds_into_video * 30
        # Play the video
        while cap.isOpened():
            ret, frame = cap.read()
            # grayscale it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if count == numFrames:
                # Crop image using parameters
                # Get top score from arguments
                # cv2.imwrite("vid4 scsh.jpg", frame)
                topframe = frame[topy:topyplush, topx:topxplusw]
                # topframe = self.crop_image_timer(frame)
                self.ocr(topframe, "timer")
                # Get bottom score
                # botframe = frame[boty:botyplush, botx:botxplusw]
                # self.ocr(botframe, "stats")
                count = 0
                numFrames = 30
            else:
                count += 1
            # print("Count", count, "numframes:", numFrames)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def run_new_video(vid, bar, era):
    stats = getShowMatchStats()
    stats.editvideo(vid, bar, era)


if __name__ == '__main__':
    stats = getShowMatchStats()
    procs = []
    proc = Process(target=)
    # stats.editvideo('Videos/Kronovi vs Scrub Killa 1v1 part 2.mp4', "bar", 1)

    # stats.get_ocr('Videos/test_video2.mp4', 40, 290, 320, 820, 855)
    # frame = cv2.imread("winner.jpg")

    # winframe = frame[290:338, 555:725]
    # stats.ocr(winframe, "winner")
    # cv2.imwrite("added.jpg", final)
    # botgoalframe = frame[395:425, 830:865]
    # stats.ocr(botgoalframe, "stats")

    # Test algorithm on more videos from different series
    # Get double digit numbers in stats working
    # Figure out how to play videos faster/download lots of videos at once
