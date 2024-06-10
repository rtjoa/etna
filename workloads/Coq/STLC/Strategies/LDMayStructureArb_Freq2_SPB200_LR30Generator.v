From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 90
      | (0, 3, 5) => 75
      | (0, 3, 6) => 81
      | (0, 5, 3) => 88
      | (0, 5, 5) => 65
      | (0, 5, 6) => 60
      | (0, 6, 3) => 90
      | (0, 6, 5) => 60
      | (0, 6, 6) => 75
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 82
      | (0, 3, 6) => 74
      | (0, 5, 3) => 0
      | (0, 5, 5) => 54
      | (0, 5, 6) => 61
      | (0, 6, 3) => 0
      | (0, 6, 5) => 85
      | (0, 6, 6) => 80
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 51
      | (1, 3, 5) => 31
      | (1, 3, 6) => 31
      | (1, 5, 3) => 49
      | (1, 5, 5) => 30
      | (1, 5, 6) => 62
      | (1, 6, 3) => 27
      | (1, 6, 5) => 24
      | (1, 6, 6) => 34
      | (2, 3, 3) => 3
      | (2, 3, 5) => 8
      | (2, 3, 6) => 7
      | (2, 5, 3) => 60
      | (2, 5, 5) => 50
      | (2, 5, 6) => 48
      | (2, 6, 3) => 2
      | (2, 6, 5) => 10
      | (2, 6, 6) => 0
      | (3, 3, 3) => 42
      | (3, 3, 5) => 50
      | (3, 3, 6) => 42
      | (3, 5, 3) => 0
      | (3, 5, 5) => 21
      | (3, 5, 6) => 0
      | (3, 6, 3) => 47
      | (3, 6, 5) => 51
      | (3, 6, 6) => 47
      | (4, 0, 3) => 17
      | (4, 0, 5) => 0
      | (4, 0, 6) => 18
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 10
      | (1, 3, 6) => 48
      | (1, 5, 3) => 1
      | (1, 5, 5) => 45
      | (1, 5, 6) => 42
      | (1, 6, 3) => 0
      | (1, 6, 5) => 37
      | (1, 6, 6) => 39
      | (2, 3, 3) => 0
      | (2, 3, 5) => 12
      | (2, 3, 6) => 1
      | (2, 5, 3) => 11
      | (2, 5, 5) => 41
      | (2, 5, 6) => 42
      | (2, 6, 3) => 0
      | (2, 6, 5) => 18
      | (2, 6, 6) => 2
      | (3, 3, 3) => 42
      | (3, 3, 5) => 46
      | (3, 3, 6) => 56
      | (3, 5, 3) => 0
      | (3, 5, 5) => 22
      | (3, 5, 6) => 0
      | (3, 6, 3) => 47
      | (3, 6, 5) => 49
      | (3, 6, 6) => 54
      | (4, 0, 3) => 17
      | (4, 0, 5) => 0
      | (4, 0, 6) => 68
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 91
      | (1, 3, 5) => 84
      | (1, 3, 6) => 89
      | (1, 5, 3) => 81
      | (1, 5, 5) => 71
      | (1, 5, 6) => 63
      | (1, 6, 3) => 89
      | (1, 6, 5) => 65
      | (1, 6, 6) => 89
      | (2, 3, 3) => 18
      | (2, 3, 5) => 30
      | (2, 3, 6) => 17
      | (2, 5, 3) => 61
      | (2, 5, 5) => 55
      | (2, 5, 6) => 53
      | (2, 6, 3) => 14
      | (2, 6, 5) => 41
      | (2, 6, 6) => 10
      | (3, 3, 3) => 64
      | (3, 3, 5) => 60
      | (3, 3, 6) => 57
      | (3, 5, 3) => 2
      | (3, 5, 5) => 60
      | (3, 5, 6) => 2
      | (3, 6, 3) => 57
      | (3, 6, 5) => 53
      | (3, 6, 6) => 50
      | (4, 0, 3) => 84
      | (4, 0, 5) => 0
      | (4, 0, 6) => 70
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 88
      | (1, 3, 5) => 70
      | (1, 3, 6) => 85
      | (1, 5, 3) => 65
      | (1, 5, 5) => 47
      | (1, 5, 6) => 30
      | (1, 6, 3) => 92
      | (1, 6, 5) => 81
      | (1, 6, 6) => 85
      | (2, 3, 3) => 94
      | (2, 3, 5) => 88
      | (2, 3, 6) => 93
      | (2, 5, 3) => 62
      | (2, 5, 5) => 53
      | (2, 5, 6) => 57
      | (2, 6, 3) => 95
      | (2, 6, 5) => 85
      | (2, 6, 6) => 94
      | (3, 3, 3) => 47
      | (3, 3, 5) => 43
      | (3, 3, 6) => 44
      | (3, 5, 3) => 95
      | (3, 5, 5) => 76
      | (3, 5, 6) => 95
      | (3, 6, 3) => 48
      | (3, 6, 5) => 47
      | (3, 6, 6) => 48
      | (4, 0, 3) => 36
      | (4, 0, 5) => 96
      | (4, 0, 6) => 26
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
