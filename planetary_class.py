import os
import re
import shutil
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class planetary_resonance:
    """
    Wrapper for Fortran planetary resonance routines (Gallardo et al. 2020).

    Calculates equilibrium points, libration periods, resonance widths,
    and Hamiltonian contours for a k2:k1 mean-motion resonance between an
    interior and an exterior planet.

    Conventions
    ----------
    - Angles (sigma, i, nodo, varpi) are in degrees.
    - Masses are in solar masses.
    - a_inn is fixed; a_out is solved for internally by the Fortran code
      from the resonance condition (hence it is written as 0.0 in the
      input file — this is intentional, not a bug).

    Each instance runs in its own temporary working directory so that
    multiple instances (e.g. when scanning parameters to compute a
    resonance width) never clobber each other's input/output files.
    """

    # Cache compiled binaries per (source path, mtime) so repeated
    # instantiations with the same Fortran source don't recompile.
    _exe_cache: dict = {}

    def __init__(
        self,
        star_mass: float,
        a_inn: float,
        e_inn: float,
        e_out: float,
        i_inn: float = 0.0,
        i_out: float = 0.0,
        nodo_inn: float = 0.0,
        nodo_out: float = 0.0,
        varpi_inn: float = 0.0,
        varpi_out: float = 0.0,
        mass_inn: float = 1e-3,
        mass_out: float = 1e-3,
        k2: int = 2,
        k1: int = 1,
        source: str = "hamiltplares.f",
        auto_run: bool = True,
    ):
        self.star_mass = star_mass
        self.a_inn = a_inn
        self.e_inn = e_inn
        self.e_out = e_out
        self.i_inn = i_inn
        self.i_out = i_out
        self.nodo_inn = nodo_inn
        self.nodo_out = nodo_out
        self.varpi_inn = varpi_inn
        self.varpi_out = varpi_out
        self.mass_inn = mass_inn
        self.mass_out = mass_out
        self.k2 = k2
        self.k1 = k1

        self.source = os.path.abspath(source)
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Fortran source not found: {self.source}")

        self._data = None
        self._workdir = tempfile.mkdtemp(prefix="planres_")

        if auto_run:
            self.run()

    # ------------------------------------------------------------------
    # Context manager support, so temp dirs get cleaned up reliably:
    #   with planetary_resonance(...) as pr: ...
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()
        
    def get_a_inn(self):
        """Return the semimajor axis of the inner planet."""
        return self.a_inn

    def get_a_out(self):
        """
        Calculate and return the resonant semimajor axis of the outer planet.
    
        The resonance condition is
    
            n_in / n_out = k2 / k1
    
        including the masses of both planets:
    
            a_out = a_inn * [
                (M_star + m_out) / (M_star + m_in)
                * (k1 / k2)^2
            ]^(1/3)
    
        Returns
        -------
        float
            Resonant outer planet semimajor axis in AU.
        """
        a_out = self.a_inn * (
            (self.star_mass + self.mass_out)
            / (self.star_mass + self.mass_inn)
            * (self.k1 / self.k2)**2
        )**(1.0 / 3.0)
    
        return a_out

    def cleanup(self):
        """Removes the instance's temporary working directory."""
        workdir = getattr(self, "_workdir", None)
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
            self._workdir = None

    def _path(self, filename: str) -> str:
        return os.path.join(self._workdir, filename)

    def _require_data(self):
        if self._data is None:
            raise RuntimeError(
                "No results available yet — call run() first "
                "(or check that the last run() succeeded)."
            )

    # ------------------------------------------------------------------
    # Input / compilation / execution
    # ------------------------------------------------------------------
    def _write_input(self):
        """Writes input configuration file for Fortran execution."""
        with open(self._path("plasysj.inp"), "w") as file:
            file.write("STAR: mass of the star in solar masses\n")
            file.write(f"{self.star_mass}\n")
            file.write("PLANETS interior and exterior: a e i lonod loper mass\n")
            file.write(
                " ".join(map(str, [self.a_inn, self.e_inn, self.i_inn,
                                    self.nodo_inn, self.varpi_inn, self.mass_inn])) + "\n"
            )
            file.write(
                " ".join(map(str, [0.0, self.e_out, self.i_out,
                                    self.nodo_out, self.varpi_out, self.mass_out])) + "\n"
            )
            file.write("semimajor axis of exterior planet defined by resonance\n")

    def compile(self) -> str:
        """
        Compiles the Fortran source with gfortran, reusing a cached binary
        if this exact source file (by path + mtime) was already compiled.
        """
        mtime = os.path.getmtime(self.source)
        cache_key = (self.source, mtime)

        cached_exe = self._exe_cache.get(cache_key)
        if cached_exe and os.path.exists(cached_exe):
            return cached_exe

        output_exe = self._path("hamiltplares.exe")
        result = subprocess.run(
            ["gfortran", self.source, "-o", output_exe],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gfortran compilation failed:\n{result.stderr}")

        self._exe_cache[cache_key] = output_exe
        return output_exe

    def run(self):
        """Executes the binary and parses the resulting output files."""
        self._write_input()
        exe_path = self.compile()

        for fname in ("resonance.dat", "rsigma.dat", "hamiltplares.dat"):
            fpath = self._path(fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        input_data = f"{self.k2}\n{self.k1}\n"
        result = subprocess.run(
            [exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            input=input_data,
            cwd=self._workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Fortran code failed:\n{result.stderr}")

        self._data = self._parse_outputs()

    def _parse_outputs(self) -> dict:
        # --- resonance.dat ---
        resonance_path = self._path("resonance.dat")
        try:
            with open(resonance_path, "r") as file:
                summary_text = file.read()
        except FileNotFoundError:
            raise RuntimeError(f"Expected output file missing: {resonance_path}")

        # --- rsigma.dat ---
        # Column layout confirmed against hamiltplares.f's
        # FORMAT(F7.2,E16.8,2A10,E16.8):
        #   [0:7]   sigma (deg)
        #   [7:23]  R - R_min
        #   [23:33] close-encounter flag ("CLOSE ENC" or blank)
        #   [33:43] equilibrium type ("E. STABLE" / "UNSTABLE" / blank)
        #   [43:59] libration period in years (only written for stable points)
        sigma_list, rmin_list = [], []
        sigma_eq, type_eq, period_eq = [], [], []

        rsigma_path = self._path("rsigma.dat")
        try:
            with open(rsigma_path, "r") as file:
                lines = file.readlines()[1:]
        except FileNotFoundError:
            raise RuntimeError(f"Expected output file missing: {rsigma_path}")

        for line in lines:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                sigma = float(line[0:7])
                rmin = float(line[7:23])
            except ValueError:
                continue  # skip malformed lines instead of crashing the whole run

            eq_type = line[33:43].strip() if len(line) > 33 else ""
            period_str = line[43:59].strip() if len(line) > 43 else ""

            sigma_list.append(sigma)
            rmin_list.append(rmin)

            if eq_type:
                sigma_eq.append(sigma)
                type_eq.append(eq_type)
                period_eq.append(float(period_str) if period_str else np.nan)

        # --- hamiltplares.dat ---
        a_vals, sig_vals, h_vals = [], [], []
        hamilt_path = self._path("hamiltplares.dat")
        try:
            with open(hamilt_path, "r") as file:
                for line in file:
                    parts = line.split()
                    if not parts:
                        continue
                    a_vals.append(float(parts[0]))
                    sig_vals.append(float(parts[1]))
                    h_vals.append(float(parts[2]))
        except FileNotFoundError:
            raise RuntimeError(f"Expected output file missing: {hamilt_path}")

        a_unique = np.array(sorted(set(a_vals)))
        sig_unique = np.array(sorted(set(sig_vals)))
        try:
            H = np.array(h_vals).reshape(len(a_unique), len(sig_unique))
        except ValueError:
            raise RuntimeError(
                "hamiltplares.dat does not form a regular (a, sigma) grid; "
                "cannot reshape into a 2D Hamiltonian array."
            )

        return {
            'sigma': np.array(sigma_list),
            'rmin': np.array(rmin_list),
            'sigma_eq': np.array(sigma_eq),
            'type_eq': np.array(type_eq),
            'period_eq': np.array(period_eq),
            'a_grid': a_unique,
            'sig_grid': sig_unique,
            'H': H,
            'summary': summary_text,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def sigmas(self) -> pd.DataFrame:
        """Returns resonant angle values and disturbing function values."""
        self._require_data()
        return pd.DataFrame({
            'sigma_deg': self._data['sigma'],
            'R_minus_Rmin': self._data['rmin'],
        })

    def equilibrium_points(self) -> pd.DataFrame:
        """Returns equilibrium angles, stability type, and libration periods."""
        self._require_data()
        return pd.DataFrame({
            'sigma_deg': self._data['sigma_eq'],
            'type': self._data['type_eq'],
            'period_yr': self._data['period_eq'],
        })
    
    def stable_eq(self) -> list[float]:
        """Returns a list of sigma angles (in degrees) for stable equilibrium points."""
        df = self.equilibrium_points()
        
        # Filter rows where type contains 'STABLE' but excludes 'UNSTABLE'
        stable_mask = df['type'].astype(str).str.contains('STABLE') & ~df['type'].astype(str).str.contains('UNSTABLE')
        
        stable_sigmas = df.loc[stable_mask, 'sigma_deg'].tolist()
        return stable_sigmas

    def periods(self) -> np.ndarray:
        """Returns libration periods (in years) for stable equilibrium points."""
        self._require_data()
        return self._data['period_eq'][~np.isnan(self._data['period_eq'])]

    def hamiltonian(self) -> dict:
        """Returns 2D grid coordinates (a, sigma) and Hamiltonian values H."""
        self._require_data()
        return {
            'a': self._data['a_grid'],
            'sigma': self._data['sig_grid'],
            'H': self._data['H'],
        }

    _WIDTH_RE = re.compile(
        r"stable full width Da1,\s*Da2\s*=\s*"
        r"([-+0-9.dDeE]+)\s+([-+0-9.dDeE]+)"
    )

    @staticmethod
    def _fortran_float(token: str) -> float:
        """Converts a Fortran-style real literal (D or E exponent) to float."""
        return float(token.replace("D", "E").replace("d", "e"))

    def width(self) -> dict:
        """
        Returns the full stable resonance width in semimajor axis (au) for
        each planet.

        hamiltplares.f already computes this internally (ANCHA1, ANCHA2) and
        writes it into resonance.dat's summary block as the line:
            "stable full width Da1, Da2=  <ANCHA1>  <ANCHA2>"
        so this simply parses it out of `self._data['summary']` rather than
        re-deriving it — no extra Fortran run required.

        ANCHA1 corresponds to the interior planet (a_inn), ANCHA2 to the
        exterior planet (a_out), matching this class's convention where
        a_inn is fixed and a_out is solved from the resonance condition.

        Returns
        -------
        dict with:
          'delta_a_inn' : full stable width in a for the interior planet (au)
          'delta_a_out' : full stable width in a for the exterior planet (au)

        Note: if the resonance is undetectable for these parameters, the
        Fortran code still writes this line with both widths equal to 0.0
        rather than raising an error.
        """
        self._require_data()
        match = self._WIDTH_RE.search(self._data['summary'])
        if not match:
            raise RuntimeError(
                "Could not find the 'stable full width Da1, Da2=' line in "
                "resonance.dat's summary. This means either the source "
                "compiled here isn't hamiltplares.f (which writes that "
                "line), or its output format has changed."
            )
        return {
            'delta_a_inn': self._fortran_float(match.group(1)),
            'delta_a_out': self._fortran_float(match.group(2)),
        }

    def plot_hamiltonian(self, num_levels: int = 20, cmap: str = 'Blues', figsize=(7, 5), plot: bool = True):
            """Plots Hamiltonian level curves or returns plotting data if plot is False."""
            self._require_data()
            a = self._data['a_grid']
            sigma = self._data['sig_grid']
            H = self._data['H']
            SIGMA, A = np.meshgrid(sigma, a)
    
            if not plot:
                return {'SIGMA': SIGMA, 'A': A, 'H': H}
    
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.contour(SIGMA, A, H, levels=num_levels, cmap=cmap)
            ax.set_xlabel(r'$\sigma\ [deg]$', fontsize=14)
            ax.set_ylabel(r'$a\ [\mathrm{au}]$', fontsize=14)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.set_title(
                rf'Resonance {self.k2}:{self.k1} '
                rf'($e_\mathrm{{in}}={self.e_inn}$, $e_\mathrm{{out}}={self.e_out}$, '
                rf'$\Delta\varpi={abs(self.varpi_inn - self.varpi_out)}$)'
            )
            plt.tight_layout()
            plt.show()

    def summary(self) -> str:
        """Returns raw summary string from resonance.dat."""
        self._require_data()
        return self._data['summary']