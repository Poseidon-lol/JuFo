from __future__ import annotations

import contextlib
import logging
import math
import re
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from rdkit import Chem

from .config import QuantumTaskConfig
from .geometry import GeometryResult

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an external QC program fails."""


@dataclass
class ProgramResult:
    properties: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[str] = None


class QuantumProgramExecutor(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        ...


class ExternalProgramExecutor:
    """Base helper for external QC engines."""

    name: str = "external"
    executable: str = ""
    work_dir: Optional[Path] = None
    cleanup: bool = True

    def __init__(self, command: Optional[str] = None) -> None:
        if command is not None:
            self.executable = command

    # -- public API -----------------------------------------------------
    def is_available(self) -> bool:
        return bool(shutil.which(self.executable))

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        if not self.is_available():
            raise ExecutionError(f"Executable '{self.executable}' not found on PATH.")
        cleanup = getattr(task, "cleanup_workdir", getattr(self, "cleanup", True))
        base_dir = getattr(task, "work_dir", None) or self.work_dir
        tmp_ctx = None
        if base_dir:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix=f"{self.name}_", dir=base_dir))
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{self.name}_")
            workdir = Path(tmp_ctx.__enter__())
        try:
            props_str = ",".join(map(str, tuple(task.properties or ())))
            logger.info(
                "QC executor %s start | method=%s basis=%s charge=%d mult=%d props=%s workdir=%s",
                self.name,
                task.method,
                task.basis,
                int(task.charge),
                int(task.multiplicity),
                props_str if props_str else "-",
                workdir,
            )
            started = time.perf_counter()
            input_path = workdir / "input.inp"
            output_path = workdir / "output.out"
            self._write_input(input_path, geometry, task)
            self._execute(task, input_path, output_path, workdir)
            properties, metadata = self._parse_output(output_path, task)
            metadata["engine"] = self.name
            metadata["method"] = task.method
            metadata["basis"] = task.basis
            metadata["level_of_theory"] = task.level_of_theory if hasattr(task, "level_of_theory") else f"{task.method}/{task.basis}"
            metadata["workdir"] = str(workdir)
            raw_text = output_path.read_text(encoding="utf-8", errors="replace")
            logger.info(
                "QC executor %s done in %.1fs | parsed_properties=%s",
                self.name,
                time.perf_counter() - started,
                ",".join(sorted(properties.keys())) if properties else "-",
            )
            return ProgramResult(properties=properties, metadata=metadata, raw_output=raw_text)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.__exit__(None, None, None)
            elif cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    # -- hooks ----------------------------------------------------------
    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        raise NotImplementedError

    def _execute(self, task: QuantumTaskConfig, input_path: Path, output_path: Path, workdir: Path) -> None:
        env = dict(os.environ)
        if task.environment:
            env.update(task.environment)
        exe_parent = Path(self.executable).parent
        path_sep = os.pathsep
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{exe_parent}{path_sep}{existing_path}" if existing_path else str(exe_parent)
        cmd = [self.executable, input_path.name]
        logger.debug("Running %s with command: %s", self.name, cmd)
        started = time.perf_counter()
        try:
            with output_path.open("w", encoding="utf-8") as out_f:
                completed = subprocess.run(
                    cmd,
                    cwd=workdir,
                    check=True,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env if env else None,
                    timeout=task.walltime_limit,
                )
        except FileNotFoundError as exc:  # pragma: no cover - depends on system
            raise ExecutionError(f"{self.executable} not found: {exc} (workdir={workdir})") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"{self.name} exceeded walltime limit ({task.walltime_limit}s) (workdir={workdir})"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr_tail = ""
            if exc.stderr:
                lines = [ln for ln in exc.stderr.strip().splitlines() if ln.strip()]
                if lines:
                    stderr_tail = lines[-1][:400]
            raise ExecutionError(
                f"{self.name} failed rc={exc.returncode} workdir={workdir} stderr_tail={stderr_tail}"
            ) from exc

        if completed.stderr:
            stderr_log = workdir / "stderr.log"
            stderr_log.write_text(completed.stderr, encoding="utf-8")
            logger.warning(
                "%s wrote stderr (%d chars) -> %s",
                self.name,
                len(completed.stderr),
                stderr_log,
            )
        logger.info(
            "QC executor %s process finished rc=%d in %.1fs",
            self.name,
            int(completed.returncode),
            time.perf_counter() - started,
        )

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError


class Psi4Executor(ExternalProgramExecutor):
    name = "psi4"
    executable = "psi4"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        coords = ""
        if geometry.xyz:
            lines = geometry.xyz.splitlines()[2:]
            coords = "\n".join(line for line in lines if line.strip())
        props = ", ".join(f"'{p}'" for p in task.properties)
        threads = 1
        try:
            import multiprocessing

            threads = max(1, multiprocessing.cpu_count() // 2)
        except Exception:
            threads = 1

        template = """memory 2 GB
set_num_threads {threads}
set scf_type df
set basis {basis}
set reference {reference}
molecule {{
{charge} {multiplicity}
{coords}
}}
set {{
    basis {basis}
}}
energy('{method}')
properties('{method}', properties=[{props}])
"""
        script = template.format(
            threads=threads,
            basis=task.basis,
            reference="uhf" if task.multiplicity != 1 else "rhf",
            charge=task.charge,
            multiplicity=task.multiplicity,
            coords=coords,
            method=task.method,
            props=props,
        )
        path.write_text(script.strip(), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        for line in text.splitlines():
            if "Total Energy =" in line:
                try:
                    energy = float(line.split()[-2])
                    metadata["total_energy"] = energy
                except Exception:
                    continue
            if "Dipole Moment" in line and "Debye" in line:
                try:
                    dipole = float(line.split()[-2])
                    props["dipole"] = dipole
                except Exception:
                    continue
        return props, metadata


class OrcaExecutor(ExternalProgramExecutor):
    name = "orca"
    executable = "orca"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        task_keywords = dict(task.keywords or {})

        def _as_pos_int(value: Any, default: Optional[int] = None) -> Optional[int]:
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                return default
            return ivalue if ivalue > 0 else default

        keywords = [task.method, task.basis, "TightSCF"]
        extra_keywords = task_keywords.get("extra_keywords")
        if isinstance(extra_keywords, str):
            extra_keywords = [extra_keywords]
        if isinstance(extra_keywords, (list, tuple)):
            for kw in extra_keywords:
                if kw:
                    keywords.append(str(kw))
        if task.solvent_model:
            if task.solvent:
                keywords.append(f"{task.solvent_model}({task.solvent})")
            else:
                keywords.append(task.solvent_model)
        use_tddft = any(
            p.lower().startswith("lambda")
            or p.lower().startswith("absorption")
            or "oscillator" in p.lower()
            for p in task.properties
        )
        use_polar = any(p.lower() == "polarizability" for p in task.properties)
        header = "! " + " ".join(keywords)
        if task.dispersion:
            header += f" {task.dispersion}"
        lines = [header]
        maxcore = _as_pos_int(task_keywords.get("maxcore"))
        if maxcore is not None:
            lines.append(f"%maxcore {maxcore}")
        nprocs = _as_pos_int(task_keywords.get("nprocs"))
        if nprocs is None:
            nprocs = _as_pos_int(task_keywords.get("pal_nprocs"))
        if nprocs is not None:
            lines.extend(["%pal", f"nprocs {nprocs}", "end"])
        if use_tddft:
            nroots = _as_pos_int(task_keywords.get("tddft_nroots"), default=10)
            lines.extend(["%tddft", f"nroots {nroots}", "end"])
        if use_polar:
            lines.extend(["%elprop", "Polar 1", "end"])
        logger.info(
            "ORCA input setup | nprocs=%s maxcore=%s tddft=%s polar=%s",
            nprocs if nprocs is not None else 1,
            maxcore if maxcore is not None else "default",
            bool(use_tddft),
            bool(use_polar),
        )
        lines.append(f"* xyz {task.charge} {task.multiplicity}")
        if geometry.xyz:
            for row in geometry.xyz.splitlines()[2:]:
                if row.strip():
                    lines.append(row)
        lines.append("*")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        orbitals = []
        in_orbital_block = False
        in_absorption_table = False
        tddft_states = []
        polar = None
        warning_lines = 0
        metadata["terminated_normally"] = False
        for line in text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if "ORCA TERMINATED NORMALLY" in upper:
                metadata["terminated_normally"] = True
            if "SCF CONVERGED AFTER" in upper:
                metadata["scf_converged"] = True
            if "SCF NOT CONVERGED" in upper or "SCF FAILED TO CONVERGE" in upper:
                metadata["scf_converged"] = False
            if "WARNING" in upper:
                warning_lines += 1
            if "TOTAL SCF ENERGY" in upper:
                continue
            if "FINAL SINGLE POINT ENERGY" in upper:
                match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?)", stripped, flags=re.I)
                if match:
                    try:
                        metadata["total_energy"] = float(match.group(1))
                    except Exception:
                        pass
            if "TOTAL ENERGY" in upper and ":" in line and "EH" in upper:
                tokens = line.replace(":", " ").split()
                for token in tokens:
                    try:
                        metadata["total_energy"] = float(token)
                        break
                    except Exception:
                        continue
            if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in upper:
                in_absorption_table = True
                continue
            if in_absorption_table and "ABSORPTION SPECTRUM VIA" in upper and "ELECTRIC DIPOLE MOMENTS" not in upper:
                in_absorption_table = False
            if "MAGNITUDE" in upper and "DEBYE" in upper:
                matches = re.findall(r"(-?\d+(?:\.\d+)?)", stripped)
                if matches:
                    with contextlib.suppress(Exception):
                        props["dipole"] = float(matches[-1])
            elif "TOTAL DIPOLE MOMENT" in upper:
                # ORCA variants can print dipole as vector components on this line.
                matches = re.findall(r"(-?\d+(?:\.\d+)?)", stripped)
                if matches:
                    with contextlib.suppress(Exception):
                        if len(matches) >= 3:
                            x, y, z = map(float, matches[-3:])
                            props["dipole"] = float(math.sqrt(x * x + y * y + z * z))
                        else:
                            props["dipole"] = float(matches[-1])
            if "ISOTROPIC POLARIZABILITY" in upper or "ISOTROPIC POLARISABILITY" in upper:
                matches = re.findall(r"(-?\d+(?:\.\d+)?)", stripped)
                if matches:
                    with contextlib.suppress(Exception):
                        polar = float(matches[-1])
                        props["polarizability"] = polar
            if "ISOTROPIC ALPHA" in upper:
                matches = re.findall(r"(-?\d+(?:\.\d+)?)", stripped)
                if matches:
                    with contextlib.suppress(Exception):
                        polar = float(matches[-1])
                        props["polarizability"] = polar
            if in_absorption_table:
                # ORCA absorption table rows can vary (e.g. "1 -> 2 ..." vs "1 ...").
                # Find the first contiguous (eV, cm^-1, nm, fosc) tuple by value ranges.
                num_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", stripped)
                if len(num_tokens) >= 4:
                    values: list[float] = []
                    for token in num_tokens:
                        with contextlib.suppress(Exception):
                            values.append(float(token))
                    if len(values) >= 4:
                        parsed = None
                        for i in range(0, len(values) - 3):
                            ev_val, cm_val, nm_val, f_val = values[i : i + 4]
                            if not (0.05 <= ev_val <= 20.0):
                                continue
                            if not (500.0 <= cm_val <= 100000.0):
                                continue
                            if not (100.0 <= nm_val <= 5000.0):
                                continue
                            if not (0.0 <= f_val <= 10.0):
                                continue
                            parsed = (ev_val, nm_val, f_val)
                            break
                        if parsed is not None:
                            tddft_states.append({"ev": parsed[0], "nm": parsed[1], "f": parsed[2]})
            is_excited_state_line = (
                ("EXCITED STATE" in upper and ("EV" in upper or "NM" in upper))
                or bool(re.match(r"^STATE\s+\d+.*(?:EV|NM)", upper))
            )
            if is_excited_state_line:
                # Accept ORCA format variants:
                #   Excited State 1: ... 3.12 eV 397.5 nm f=0.1234
                #   STATE 1: E=3.12 eV  fosc=0.1234
                ev_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*eV", stripped, flags=re.I)
                nm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", stripped, flags=re.I)
                f_match = re.search(
                    r"(?:f(?:osc)?\s*=\s*|oscillator\s*strength\s*[:=]\s*)([0-9]+(?:\.[0-9]+)?)",
                    stripped,
                    flags=re.I,
                )
                try:
                    state = {
                        "ev": float(ev_match.group(1)) if ev_match else None,
                        "nm": float(nm_match.group(1)) if nm_match else None,
                        "f": float(f_match.group(1)) if f_match else None,
                    }
                    if state["nm"] is None and state["ev"] is not None and state["ev"] > 1e-12:
                        state["nm"] = 1239.841984 / state["ev"]
                    if state["ev"] is not None or state["nm"] is not None:
                        tddft_states.append(state)
                except Exception:
                    continue
            if stripped.startswith("ORBITAL ENERGIES"):
                in_orbital_block = True
                continue
            if in_orbital_block:
                if not stripped or stripped.startswith("---") or stripped.startswith("NO"):
                    continue
                tokens = stripped.split()
                if len(tokens) >= 4 and tokens[0].isdigit():
                    try:
                        occ = float(tokens[1])
                        energy_ev = float(tokens[3])
                        orbitals.append((occ, energy_ev))
                        continue
                    except Exception:
                        pass
                # end of block when non-parsable line encountered after data started
                if orbitals:
                    in_orbital_block = False
        occupied = [energy for occ, energy in orbitals if occ > 0.0]
        virtual = [energy for occ, energy in orbitals if occ <= 0.0]
        if occupied:
            props["HOMO"] = occupied[-1]
        if virtual:
            props["LUMO"] = virtual[0]
        if "HOMO" in props and "LUMO" in props:
            props["gap"] = props["LUMO"] - props["HOMO"]
        # TDDFT excitations: choose strongest oscillator if available, else first
        if tddft_states:
            with_f = [s for s in tddft_states if s.get("f") is not None]
            best = max(with_f, key=lambda s: s["f"]) if with_f else tddft_states[0]
            if best.get("nm") is not None:
                props["lambda_max_nm"] = float(best["nm"])
            if best.get("f") is not None:
                props["oscillator_strength"] = float(best["f"])
        metadata["warning_lines"] = int(warning_lines)
        logger.info(
            "ORCA parse summary | terminated=%s scf=%s E=%s HOMO=%s LUMO=%s gap=%s lambda_max_nm=%s f=%s warnings=%d",
            metadata.get("terminated_normally"),
            metadata.get("scf_converged", "unknown"),
            metadata.get("total_energy", "n/a"),
            props.get("HOMO", "n/a"),
            props.get("LUMO", "n/a"),
            props.get("gap", "n/a"),
            props.get("lambda_max_nm", "n/a"),
            props.get("oscillator_strength", "n/a"),
            int(warning_lines),
        )
        return props, metadata


class GaussianExecutor(ExternalProgramExecutor):
    name = "gaussian"
    executable = "g16"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        header = f"%Chk=job.chk\n#P {task.method}/{task.basis} Pop=Full"
        if task.dispersion:
            header += f" EmpiricalDispersion={task.dispersion}"
        lines = [header, "", "Generated by qc.executor", "", f"{task.charge} {task.multiplicity}"]
        if geometry.xyz:
            for row in geometry.xyz.splitlines()[2:]:
                if row.strip():
                    lines.append(row)
        lines.append("")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        for line in text.splitlines():
            if "SCF Done:" in line:
                try:
                    metadata["total_energy"] = float(line.split()[4])
                except Exception:
                    continue
        return props, metadata


class SemiEmpiricalExecutor(QuantumProgramExecutor):
    """Fallback executor using RDKit-derived heuristics."""

    name = "semi_empirical"

    def is_available(self) -> bool:
        return True

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = geometry.mol or Chem.MolFromSmiles(geometry.smiles)  # type: ignore[name-defined]
        if mol is None:
            raise ExecutionError("Cannot construct RDKit molecule for surrogate executor.")

        mw = Descriptors.MolWt(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Crippen.MolLogP(mol)
        frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        num_rot = Descriptors.NumRotatableBonds(mol)

        # crude heuristics for demonstration / fast fallback
        homo = -4.8 - 0.002 * (mw - 300) - 0.18 * aromatic + 0.05 * frac_csp3
        lumo = homo + 2.2 - 0.004 * rings + 0.1 * frac_csp3
        gap = lumo - homo
        ie = -homo
        ea = -lumo
        lam_hole = 0.22 + 0.005 * num_rot + 0.015 * aromatic
        lam_electron = 0.25 + 0.004 * rings + 0.01 * tpsa / 100
        dipole = 1.5 + 0.02 * tpsa
        polar = 3.0 + 0.1 * mw / 100
        packing = max(0.0, 1.0 - (tpsa / 150 + abs(logp) / 7.0))
        stability = 0.5 * ie + 0.2 * polar - 0.1 * lam_hole

        props = {
            "HOMO": homo,
            "LUMO": lumo,
            "gap": gap,
            "IE": ie,
            "EA": ea,
            "lambda_hole": lam_hole,
            "lambda_electron": lam_electron,
            "dipole": dipole,
            "polarizability": polar,
            "packing_score": packing,
            "stability_index": stability,
        }
        filtered = {k: v for k, v in props.items() if k in task.properties or not task.properties}
        metadata = {
            "engine": self.name,
            "heuristic": True,
            "mw": mw,
            "rings": rings,
            "aromatic_rings": aromatic,
            "tpsa": tpsa,
            "logp": logp,
        }
        return ProgramResult(properties=filtered, metadata=metadata)


DEFAULT_EXECUTORS: Mapping[str, QuantumProgramExecutor] = {
    "psi4": Psi4Executor(),
    "gaussian": GaussianExecutor(),
    "orca": OrcaExecutor(),
    "semi_empirical": SemiEmpiricalExecutor(),
}


def resolve_executor(name: str | None) -> QuantumProgramExecutor:
    if not name:
        return DEFAULT_EXECUTORS["semi_empirical"]
    lowered = name.lower()
    if lowered in DEFAULT_EXECUTORS:
        return DEFAULT_EXECUTORS[lowered]
    raise KeyError(f"Unbekannter QC executor '{name}'")
