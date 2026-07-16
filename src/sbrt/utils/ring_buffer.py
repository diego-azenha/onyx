"""Buffer circular O(1) — usado por accumulators.py (janelas rodantes) e h0.py (lags do AR,
atravessando a fronteira histórico->online, plano §3.1 item 8 / armadilha §13.3)."""
from __future__ import annotations


class RingBuffer:
    """Capacidade fixa. `push` é O(1); `peek(age)` lê o valor `age` passos atrás sem alocar.

    Convenção de idade: logo após um `push(x)`, `peek(0) == x` (o mais recente), `peek(1)` é o
    penúltimo, etc. Para janelas rodantes de tamanho w, o valor que *sai* da janela ao inserir uma
    nova observação é `peek(w - 1)` chamado ANTES do push (ver state/accumulators.py).
    """

    __slots__ = ("capacity", "_buf", "_head", "_count")

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity deve ser >= 1")
        self.capacity = capacity
        self._buf = [0.0] * capacity
        self._head = -1
        self._count = 0

    def push(self, x: float) -> float | None:
        """Insere x; retorna o valor expulso (se o buffer já estava cheio) ou None."""
        self._head = (self._head + 1) % self.capacity
        evicted = self._buf[self._head] if self._count >= self.capacity else None
        self._buf[self._head] = x
        if self._count < self.capacity:
            self._count += 1
        return evicted

    def peek(self, age: int) -> float:
        if age < 0 or age >= self.capacity:
            raise IndexError(f"age {age} fora de [0, {self.capacity})")
        idx = (self._head - age) % self.capacity
        return self._buf[idx]

    def __len__(self) -> int:
        return self._count
