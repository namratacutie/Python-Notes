class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []

    def add_songs(self, song):
        self.songs.append(song)
        print(f"Added songs : {song}")

    def show_songs(self):
        for song in self.songs:
            print(f"Your songs : {song}")

my_playlist = Playlist("Favorites")
my_playlist.add_songs("Iku Iku")
my_playlist.show_songs()