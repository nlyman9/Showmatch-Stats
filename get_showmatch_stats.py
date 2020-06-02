from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import sys
import uuid
from multiprocessing import Process, current_process


class getShowMatchStats:

    def __init__(self):
        self.initialendgametime = 9000  # Take the first screen capture after 5 minutes
        self.endgametime = -1  # How long is currently left on the timer
        self.initial_frame = True  # Flag for getting the initial timer capture
        self.timer_zero = False  # Flag for if the timer equals zero
        self.winner_chosen = False  # Flag for if winner has been displayed
        self.isreplay = False
        self.grabcount = True
        self.check_for_replay = False
        self.temp = 0

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

        self.game_number = 1
        self.show_ocr = False
        self.show_video = False

    # Do Optical Character Recognition on img
    def ocr(self, img, config):
        langstats = False  # Flag for the stats language
        langtimer = False  # Flag for the timer language
        langreplay = False  # Flag for the replay language

        if config == "timer":
            langtimer = True
            custom_config = r'-c tessedit_char_whitelist=0123456789: --psm 7'
        elif config == "winner":
            custom_config = r'-c tessedit_char_blacklist=., --psm 8'
        elif config == "single":
            # print("Using single digit")
            langstats = True
            custom_config = r' --psm 8 outputbase digits'
        elif config == "replay":
            langreplay = True
            custom_config = r' --psm 8'
        else:
            custom_config = r'--psm 7'
        pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract.exe'
        newImage = cv2.bitwise_not(img)
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filename = "{}.jpg".format(str(uuid.uuid4()))
        cv2.imwrite(filename, gray)
        if langstats:
            text = pytesseract.image_to_string(Image.open(filename), lang='rl', config=custom_config)
        elif langtimer:
            text = pytesseract.image_to_string(Image.open(filename), lang='langtimer', config=custom_config)
        elif langreplay:
            text = pytesseract.image_to_string(Image.open(filename), lang='replay', config=custom_config)
        else:
            text = pytesseract.image_to_string(Image.open(filename), config=custom_config)
        os.remove(filename)
        if self.show_ocr:
            self.show_ocr_result(text, gray, img)
        return text

    def editvideo(self, vid, filename, era):
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
                self.get_timer_value(frame, era)
                self.initial_frame = False
                count = 0
            elif not self.initial_frame and count == self.endgametime and not self.winner_chosen:
                # Wait however many seconds are left in game, then take screenshot of timer
                if self.timer_zero is False:
                    self.get_timer_value(frame, era)
                # Check if the timer is 0
                if self.endgametime == 0 or self.timer_zero is True or self.endgametime == 1:
                    self.timer_zero = True
                    # Check timer every second to get when the ball actually hits the ground ending the game
                    winner = self.wait_ball_hit_ground(frame, era)
                    # print("Winner is currently:", winner)
                    if winner == "winner":
                        # print("Winner winner chicken dinner")
                        self.winner_chosen = True
                        # The game has officially ended
                count = 0
            elif self.winner_chosen:
                # Wait 12 seconds for stats screen to show
                if era == 1:
                    self.endgametime = 150
                elif 2 <= era <= 4 or era == 6:
                    self.endgametime = 360
                elif era == 5:
                    self.endgametime = 300
                elif era == 7 or era == 8:
                    # Wait 9 seconds instead because shorter celebration
                    self.endgametime = 270
                else:
                    # self.endgametime = 390
                    # Try 12 seconds to grab scoreboard
                    self.endgametime = 360

                if count == self.endgametime:
                    # cv2.imwrite("era4scoreboard.jpg", frame)
                    # Get information from the stats screen
                    self.capture_stats_screen(frame, filename, era)
                    # cv2.waitKey(0)
                    # Check for 1 minute if the next game is simply a replay
                    if self.game_number > 1:
                        self.check_for_replay = True
                    self.initial_frame = True
                    self.timer_zero = False
                    self.winner_chosen = False
                    self.initialendgametime = 9000  # Check timer after 5 mins
                    self.endgametime = -1
                    count = 0
                    self.game_number += 1

            if self.check_for_replay:
                if self.grabcount:
                    self.temp = count
                    self.grabcount = False
                if count - self.temp <= 2100:   # Check for 1 minute, 10 seconds
                    # Check every second for the notice screen tkhat the rest of video is replays
                    if count % 30 == 0:
                        text = self.check_for_replay_screen(frame, era)
                        if text == "replay":
                            # The next games of the video are just replays
                            # print("Rest of the video is replay")
                            break
                else:
                    # This is a legitimate game that should be counted so reset
                    self.initial_frame = True
                    self.timer_zero = False
                    self.winner_chosen = False
                    self.initialendgametime = 7200  # Check timer after 4 mins
                    self.endgametime = -1
                    self.grabcount = True
                    self.check_for_replay = False
                    count = 0
            count += 1
            # print("Count", count, "endgametime", self.endgametime)
            if self.show_video:
                cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Crop the frame so that it just includes the timer using 720p res
    def crop_image_timer(self, frame, era):
        crop_img = -1
        if 1 <= era <= 3:
            crop_img = frame[13:42, 590:695]
        elif era == 4 or era == 5 or era == 8:
            crop_img = frame[10:35, 600:685]
        elif era == 6 or era == 7:
            crop_img = frame[25:50, 600:685]
        return crop_img

    # Get the number of frames until the end of the rocket league game at 30fps
    def calc_new_end_game(self, time):
        digitlist = []
        for digit in range(len(time)):
            if time[digit] == ':':
                # print("Digit was :, not appending")
                pass
            else:
                digitlist.append(int(time[digit]))
        if len(digitlist) == 3:
            end_game = ((digitlist[0] * 60) + (digitlist[1] * 10) + digitlist[2]) * 30
            if end_game == 0 or (end_game == 30 and digitlist[2] == 1):
                return 0
            elif self.verify_previous_timer_val == -1:
                # Set the previous timer value and set endgame = 30 to get the next second
                self.verify_previous_timer_val = end_game
                end_game = 30
                # print("Setting previous val =", self.verify_previous_timer_val, "and taking a screenshot in next second")
            else:
                # If this == 1 or 0, the timer's time is verified
                if (self.verify_previous_timer_val - end_game) == 30 or (
                        self.verify_previous_timer_val - end_game) == 0:
                    # Verified
                    # print("Subtraction equals 0 or 1 so timer is verified")
                    self.verify_previous_timer_val = -1
                else:
                    # Timer is not verified, take another timer value at next second
                    # print("Some number was messed up. Taking a screenshot in next second and setting previous =", self.verify_previous_timer_val, "to", end_game)
                    self.verify_previous_timer_val = end_game
                    end_game = 30
        else:
            # print("Digitlist is not 3, taking new screenshot after a second")
            end_game = 30  # This makes the timer wait one second before selecting again

        # Store the previous timer value into a global variable
        # Have -1 as the need to set value of the global variable and anything else means you should compare the current val to the previous val
        # To make sure it's valid make sure the previous value - the current value = 1 or 0.

        # print("Next timer check is in", (end_game / 30), "second(s)")
        # Has a chance to miss the timer hitting 0 if exact number is on the timer and no stoppages so subtract a second
        if end_game > 30:
            end_game = end_game - 30
        return end_game

    def check_for_replay_screen(self, frame, era):
        cframe = self.crop_image_replay(frame, era)
        text = self.ocr(cframe, "replay")
        return text

    def get_timer_value(self, frame, era):
        cframe = self.crop_image_timer(frame, era)
        text = self.ocr(cframe, "timer")
        self.endgametime = self.calc_new_end_game(text)

    def wait_ball_hit_ground(self, frame, era):
        cframe = self.crop_image_winner(frame, era)
        text = self.ocr(cframe, "winner")
        self.endgametime = 30
        return text.lower()

    def crop_image_stats(self, frame):
        crop_img = frame[20:65, 585:700]
        return crop_img

    def crop_image_replay(self, frame, era):
        if 5 <= era <= 7:
            frame = frame[80:190, 505:825]  # Used in vid 211
        elif era == 8:
            frame = frame[80:220, 505:825]  # Used in vid 716
        return frame

    def crop_image_winner(self, frame, era):
        crop_img = -1
        if era == 1:
            crop_img = frame[275:325, 540:740]
        elif era == 2 or era == 3:
            crop_img = frame[280:325, 555:725]
        elif era == 4 or era == 5:
            crop_img = frame[305:330, 577:703]
        elif era == 6:  # or era == 7: # This is for era 7 without the series, aka era 7 after vid
            crop_img = frame[306:332, 577:703]
        elif era == 7:
            # This is for when the winner is displayed for the series
            crop_img = frame[306:332, 635:730]
        elif era == 8:
            crop_img = frame[302:330, 590:690]
        return crop_img

    def show_ocr_result(self, text, gray, img):
        print("Text", text)
        cv2.imshow("Image", img)
        cv2.imshow("Output", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def scaleup(self, frame, scale):
        scale_percent = scale * 100
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(frame, dsize)
        return output

    # Wait from the time the text "winner" is displayed to the time the stats screen is displayed
    # Convert the stats screen to strings
    def capture_stats_screen(self, frame, filename, era):
        suffix = "{}.txt".format(self.game_number)
        filename = filename + suffix
        if 1 <= era <= 3:
            topname_dim = [285, 310, 513, 715]
            topgoals_dim = [290, 320, 820, 855]
            topassists_dim = [290, 320, 885, 925]
            topsaves_dim = [290, 320, 960, 995]
            topshots_dim = [290, 320, 1015, 1055]
            botname_dim = [390, 415, 513, 715]
            botgoals_dim = [395, 425, 820, 855]
            botassists_dim = [395, 425, 885, 925]
            botsaves_dim = [395, 425, 960, 995]
            botshots_dim = [395, 425, 1015, 1055]

            subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim,
                         botgoals_dim, botassists_dim, botsaves_dim, botshots_dim]

            self.write_to_file(frame, filename, subframes)
        elif era == 4 or era == 5:
            topname_dim = [300, 325, 545, 695]
            topgoals_dim = [300, 325, 775, 805]
            topassists_dim = [300, 325, 823, 855]
            topsaves_dim = [300, 325, 878, 905]
            topshots_dim = [300, 325, 930, 960]
            botname_dim = [375, 400, 545, 695]
            botgoals_dim = [380, 405, 775, 805]
            botassists_dim = [380, 405, 823, 855]
            botsaves_dim = [380, 405, 878, 905]
            botshots_dim = [380, 405, 930, 960]

            subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim,
                         botgoals_dim, botassists_dim, botsaves_dim, botshots_dim]

            self.write_to_file(frame, filename, subframes)
        elif era == 6:
            topname_dim = [305, 320, 550, 695]
            topgoals_dim = [307, 330, 765, 795]
            topassists_dim = [305, 330, 815, 845]
            topsaves_dim = [305, 330, 870, 895]
            topshots_dim = [305, 330, 920, 948]
            botname_dim = [378, 395, 550, 695]
            botgoals_dim = [380, 405, 765, 795]
            botassists_dim = [380, 405, 815, 845]
            botsaves_dim = [380, 405, 870, 895]
            botshots_dim = [380, 405, 920, 950]

            subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim,
                         botgoals_dim, botassists_dim, botsaves_dim, botshots_dim]

            self.write_to_file(frame, filename, subframes)
        elif era == 7:
            topname_dim = [300, 310, 525, 700]
            topgoals_dim = [300, 320, 745, 770]
            topassists_dim = [300, 320, 795, 820]
            topsaves_dim = [300, 320, 845, 870]
            topshots_dim = [300, 320, 895, 920]
            botname_dim = [383, 395, 525, 700]
            botgoals_dim = [380, 400, 745, 770]
            botassists_dim = [380, 400, 795, 820]
            botsaves_dim = [380, 400, 845, 870]
            botshots_dim = [380, 400, 895, 920]

            subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim,
                         botgoals_dim,
                         botassists_dim, botsaves_dim, botshots_dim]
            self.write_to_file(frame, filename, subframes)
        elif era == 8:
            topname_dim = [295, 310, 520, 675]
            topgoals_dim = [297, 317, 750, 775]
            topassists_dim = [297, 317, 800, 825]
            topsaves_dim = [297, 317, 855, 880]
            topshots_dim = [297, 317, 910, 935]
            botname_dim = [380, 395, 520, 675]
            botgoals_dim = [382, 402, 750, 775]
            botassists_dim = [382, 402, 800, 825]
            botsaves_dim = [382, 402, 855, 880]
            botshots_dim = [382, 402, 910, 935]
            subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim,
                         botgoals_dim,
                         botassists_dim, botsaves_dim, botshots_dim]
            self.write_to_file(frame, filename, subframes)
    # Get name for top
    def get_name(self, frame, subframe):
        name = frame[subframe[0]:subframe[1], subframe[2]:subframe[3]]
        name = self.scaleup(name, 3)
        # cv2.imwrite("topname.jpg", frame_topname)
        name = self.ocr(name, "name")
        return name

    # Get desired stat
    def get_stat(self, frame, subframe):
        stat = frame[subframe[0]:subframe[1], subframe[2]:subframe[3]]
        stat = self.scaleup(stat, 3)
        stat = self.ocr(stat, "single")
        return stat

    def write_to_file(self, frame, filename, subframes):
        newfile = open(filename, "w")
        count = 0
        for fr in subframes:
            if count == 0 or count == 5:
                info = self.get_name(frame, fr)
                newfile.write(info + ",")
            else:
                info = self.get_stat(frame, fr)
                # print("Stat: ", info)
                if count == 4:
                    newfile.write(info + "\n")
                elif count == 9:
                    newfile.write(info)
                else:
                    newfile.write(info + ",")
            count += 1
        newfile.close()

    # Helper function to do an OCR on a specific part of a frame of a video that happens after seconds_into_video seconds
    def get_ocr(self, vid, seconds_into_video, seconds_into_video2,
                seconds_into_video3):  # topy, topyplush, topx, topxplusw):
        cap = cv2.VideoCapture(vid)
        count = 0
        numFrames = seconds_into_video * 30
        numFrames2 = seconds_into_video2 * 30
        numFrames3 = seconds_into_video3 * 30
        # Play the video
        while cap.isOpened():
            ret, frame = cap.read()
            # grayscale it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if count == numFrames or count == numFrames2 or count == numFrames3:
                # Crop image using parameters
                # Get top score from arguments
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame', gray)
                # cv2.waitKey(0)
                cv2.imwrite("{}.jpg".format(str(uuid.uuid4())), frame)
                # topframe = frame[topy:topyplush, topx:topxplusw]
                # topframe = self.crop_image_timer(frame)
                # self.ocr(topframe, "timer")
                # Get bottom score
                # botframe = frame[boty:botyplush, botx:botxplusw]
                # self.ocr(botframe, "stats")
                # count = 0
                # numFrames = 30
                if count == numFrames3:
                    break
                count += 1
            else:
                count += 1
            # if count % 1000 == 0:
            print("Count", count, "numframes3:", numFrames3)
            # print("Count", count, "numframes:", numFrames)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def get_ocr_info(self, frame, subframes):
        count = 0
        for fr in subframes:
            if count == 0 or count == 5:
                info = self.get_name(frame, fr)
            else:
                info = self.get_stat(frame, fr)
            count += 1


def run_new_video(arglist):
    stats = getShowMatchStats()
    stats.editvideo(arglist[0], arglist[1], arglist[2])

# Get the era based on the video's positioning in the playlist
def get_era(num):
    if 1 <= num <= 7:
        return 1
    elif 8 <= num <= 50:
        return 2
    elif 51 <= num <= 94:
        return 3
    elif 95 <= num <= 198:
        return 4
    elif 199 <= num <= 265:
        return 5
    elif 266 <= num <= 360:
        return 6
    elif 361 <= num <= 402:
        return 7
    else:
        return 8


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Number of arguments must be 2")
        sys.exit()

    startofrange = int(sys.argv[1])
    endofrange = int(sys.argv[2])

    if endofrange - startofrange > 7:
        print("Must input a maximum range of 8 videos")
        sys.exit()

    list_of_args = []
    for i in range(startofrange, endofrange + 1):
        args = ['Videos/vid{}.mp4'.format(i), 'Results/vid{}-'.format(i), get_era(i)]
        list_of_args.append(args)

    stats = getShowMatchStats()

    procs = []
    for ls in list_of_args:
        proc = Process(target=run_new_video, args=(ls,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    # Uncomment these lines to figure out dimensions for new eras
    # stats.get_ocr('Videos/vid403.mp4', 200, 514, 528)
    # frame = cv2.imread("vid403scoreboard.jpg")
    # topname_dim = [295, 310, 520, 675]
    # topgoals_dim = [297, 317, 750, 775]
    # topassists_dim = [297, 317, 800, 825]
    # topsaves_dim = [297, 317, 855, 880]
    # topshots_dim = [297, 317, 910, 935]
    # botname_dim = [380, 395, 520, 675]
    # botgoals_dim = [382, 402, 750, 775]
    # botassists_dim = [382, 402, 800, 825]
    # botsaves_dim = [382, 402, 855, 880]
    # botshots_dim = [382, 402, 910, 935]
    #
    # subframes = [topname_dim, topgoals_dim, topassists_dim, topsaves_dim, topshots_dim, botname_dim, botgoals_dim,
    #              botassists_dim, botsaves_dim, botshots_dim]
    #
    # print("Stats")
    # stats.get_ocr_info(frame, subframes)
    #
    # print("Timer")
    # frame = cv2.imread("vid403timer.jpg")
    # subframe = frame[10:35, 600:680]
    # stats.ocr(subframe, "timer")
    #
    # print("Winner")
    # frame = cv2.imread("vid403winner.jpg")
    # subframe = frame[302:330, 590:690]
    # stats.ocr(subframe, "winner")

    # frame = cv2.imread("vid266replay.jpg")
    # subframe = frame[80:190, 505:825]
    # stats.ocr(subframe, "replay")
    #
    # frame = cv2.imread("vid716replay.jpg")
    # subframe = frame[80:230, 505:825]
    # stats.ocr(subframe, "replay")

    # Video 211 is weird because no black screen, might have to review 199 - 230 ish manually

    # Some problems are that johnny can save a replay during the time I'd normally screenshot the end of a game
    # - just do manually as impossible to predict and solution would slow alg down too much
    # Also some videos might have scoreboards that are slightly higher or lower than avg and not sure if that will be problematic
    # No more series in era 7 after vid 392. Ensuing eras do not use series

    # Era list:
    # Era 1 is for videos 1-7 that don't have end of game celebrations
    # Era 2 is for videos 8-50 with the scoreboard being at the "2" variables except vid 10 and end game celebrations
    # Era 3 is for videos 51-94 with scoreboard being a bit higher for no reason
    # Era 4 is for videos 95-198 with timer smaller, scoreboard smaller, and winner smaller
    # Era 5 is for videos 199-265 just like era 4, but search for the games being replayed if necessary. Capture the "Thanks" in thanks for watching. Do Videos 199-210 manually. Video 211 starts black background
    # Era 6 is for videos 266-360 with timer further down, winner smaller, scoreboard smaller
    # Era 7 is for videos 361-402 with shorter celebration time and different scoreboard position. Also may do weird series thing. 361 - 392 is using the series
    # Era 8 is for videos 403-717 with timer in different position and larger, scoreboard slightly larger

    # Stopped normalizing text documents at vid35. Start on vid36
