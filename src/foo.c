#include <pthread.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
// int main() {
//
//   int const WIDTH = 500;
//   int const HEIGHT = 500;
//   InitWindow(WIDTH, HEIGHT, "Neuromesh");
//
//   SetTargetFPS(60);
//
//   Image image = LoadImage("../assets/bg1.png");
//
//   Texture2D texture = LoadTextureFromImage(image);
//
//   BeginBlendMode(BLEND_ALPHA);
//   UnloadImage(image);
//   printf("raylib Version: %i.%i.%i\n", RAYLIB_VERSION_MAJOR,
//          RAYLIB_VERSION_MINOR, RAYLIB_VERSION_PATCH);
//
//   Color redish = {255, 59, 0, 100};
//   Color Blue1 = {0, 187, 255, 100};
//   Color Blue2 = {0, 71, 153, 100};
//   Color Blue3 = {0, 111, 255, 55};
//   Color Blue4 = {0, 111, 255, 25};
//   Color LightBlue = {175, 234, 255, 100};
//   Color White2 = {255, 255, 255, 20};
//   Color LightYellow = {255, 195, 0, 20};
//   Color LightRed = {255, 0, 0, 20};
//   Color OffWhite = {255, 255, 255, 20};
//   Color OffGrey = {255, 255, 255, 1};
//   Color NearWhite = {255, 255, 255, 140};
//
//   const Color BG_COLOR = {1, 1, 1, 100};
//   int frame = 0;
//   while (!WindowShouldClose()) {
//     BeginDrawing();
//     ClearBackground(BG_COLOR);
//     DrawTexture(texture, 0, 0, WHITE);
//     DrawText("NEUROMESH", 20, 20, 16, LIGHTGRAY);
//
//     // DrawCircleGradient(200, 200, 20, YELLOW, MAROON);
//     DrawCircleGradient((Vector2){WIDTH / 5.0f, 220.0f}, 60.0, GREEN, MAROON);
//     // SKYBLUE);
//     DrawFPS(10, 10);
//     EndDrawing();
//     frame++;
//   }
//
//   UnloadTexture(texture);
//   EndBlendMode();
//   CloseWindow();
//   return 0;
// }

#include "pthread.h"

typedef struct {
  pthread_mutex_t count_mutex;
  unsigned int count;
} t_counter;

void *print(void *data) {

  pthread_t id;
  id = pthread_self();
  printf("My thread id: %ld\n", id);

  t_counter *counter;
  counter = (t_counter *)data;

  pthread_mutex_lock(&counter->count_mutex);
  printf("Thread: %ld: Start Count: %u\n", id, counter->count);
  pthread_mutex_unlock(&counter->count_mutex);

  const int TIME = 1000;
  int i = 0;
  while (i < TIME) {
    pthread_mutex_lock(&counter->count_mutex);
    counter->count++;
    pthread_mutex_unlock(&counter->count_mutex);
    i++;
  }

  pthread_mutex_lock(&counter->count_mutex);
  printf("Thread: %ld: Final Count: %u\n", id, counter->count);
  pthread_mutex_unlock(&counter->count_mutex);
  return NULL;
}

int main() {
  pthread_t thread, thread2;

  t_counter counter;
  counter.count = 0;

  pthread_mutex_init(&counter.count_mutex, NULL);

  pthread_create(&thread, NULL, print, &counter);
  pthread_create(&thread2, NULL, print, &counter);

  pthread_join(thread, NULL);
  pthread_join(thread2, NULL);

  printf("final Count: %u", counter.count);
  pthread_mutex_destroy(&counter.count_mutex);
  return 0;
}
