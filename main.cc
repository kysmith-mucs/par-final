#include <Magick++.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <print>
#include <random>
#include <vector>

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

using namespace Magick;

struct Grain
{
  double x, y;
};

double
get_amplitude (double x, double y, int n, int m)
{
  return std::abs (std::sin (n * M_PI * x) * std::sin (m * M_PI * y) -
                   std::sin (m * M_PI * x) * std::sin (n * M_PI * y));
}

int
main (int argc, char** argv)
{
  MPI_Init (&argc, &argv);
  InitializeMagick (*argv);

  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  const int WIDTH { 1024 };
  const int HEIGHT { 1024 };
  const int GRAINS_PER_PROC { 150000 };
  const int ITERATIONS { 1500 };
  const int N_MODE { 4 };
  const int M_MODE { 3 };

  double local_y_min { (double) rank / size };
  double local_y_max { (double) (rank + 1) / size };

  std::vector<Grain> local_grains;
  std::mt19937 gen (42 + rank);
  std::uniform_real_distribution<double> dist_x (0.0, 1.0);
  std::uniform_real_distribution<double> dist_y (local_y_min, local_y_max);
  std::uniform_real_distribution<double> step_move (-0.01, 0.01);

  double start { MPI_Wtime () };

  for (int i = 0; i < GRAINS_PER_PROC; ++i)
  {
    local_grains.push_back ({ dist_x (gen), dist_y (gen) });
  }

  for (int iter = 0; iter < ITERATIONS; ++iter)
  {
    std::vector<std::vector<Grain>> buckets (size);

    for (auto& g : local_grains)
    {
      double current_amp = get_amplitude (g.x, g.y, N_MODE, M_MODE);

      double next_x =
        std::clamp (g.x + step_move (gen) * current_amp, 0.0, 1.0);
      double next_y =
        std::clamp (g.y + step_move (gen) * current_amp, 0.0, 1.0);

      double next_amp = get_amplitude (next_x, next_y, N_MODE, M_MODE);

      if (next_amp < current_amp)
      {
        g.x = next_x;
        g.y = next_y;
      }

      int target_rank = std::clamp ((int) (g.y * size), 0, size - 1);
      buckets[target_rank].push_back (g);
    }

    std::vector<int> send_counts (size), recv_counts (size);
    for (int i = 0; i < size; ++i)
      send_counts[i] = (int) buckets[i].size ();

    MPI_Alltoall (send_counts.data (),
                  1,
                  MPI_INT,
                  recv_counts.data (),
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> s_displs (size, 0), r_displs (size, 0);
    for (int i = 1; i < size; ++i)
    {
      s_displs[i] = s_displs[i - 1] + send_counts[i - 1];
      r_displs[i] = r_displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<Grain> send_buf;
    for (auto& b : buckets)
      send_buf.insert (send_buf.end (), b.begin (), b.end ());

    local_grains.resize (r_displs[size - 1] + recv_counts[size - 1]);

    MPI_Datatype MPI_GRAIN;
    MPI_Type_contiguous (2, MPI_DOUBLE, &MPI_GRAIN);
    MPI_Type_commit (&MPI_GRAIN);

    MPI_Alltoallv (send_buf.data (),
                   send_counts.data (),
                   s_displs.data (),
                   MPI_GRAIN,
                   local_grains.data (),
                   recv_counts.data (),
                   r_displs.data (),
                   MPI_GRAIN,
                   MPI_COMM_WORLD);

    MPI_Type_free (&MPI_GRAIN);
  }

  std::vector<uint32_t> local_canvas (WIDTH * HEIGHT, 0);
  for (const auto& g : local_grains)
  {
    int px = std::clamp ((int) (g.x * (WIDTH - 1)), 0, WIDTH - 1);
    int py = std::clamp ((int) (g.y * (HEIGHT - 1)), 0, HEIGHT - 1);
    local_canvas[py * WIDTH + px]++;
  }

  std::vector<uint32_t> global_canvas;
  if (rank == 0)
    global_canvas.resize (WIDTH * HEIGHT);

  MPI_Reduce (local_canvas.data (),
              global_canvas.data (),
              WIDTH * HEIGHT,
              MPI_UINT32_T,
              MPI_SUM,
              0,
              MPI_COMM_WORLD);

  if (rank == 0)
  {
    Image image { Geometry (WIDTH, HEIGHT), Color ("black") };

    uint32_t max_val = 0;
    for (auto v : global_canvas)
      if (v > max_val)
        max_val = v;

    for (int y = 0; y < HEIGHT; ++y)
    {
      for (int x = 0; x < WIDTH; ++x)
      {
        uint32_t count = global_canvas[y * WIDTH + x];
        if (count > 0)
        {
          double factor = std::log1p (count) / std::log1p (max_val);
          Quantum q = factor * QuantumRange;
          image.pixelColor (x, y, Color (q, q, q));
        }
      }
    }
    image.write ("chladni_result.png");
    std::println ("Pattern generated: chladni_result.png");
  }

  double end { MPI_Wtime () };
  if (rank == 0)
    std::println ("Time elapsed: {:.2f} Seconds", (end - start));

  MPI_Finalize ();
  return 0;
}
