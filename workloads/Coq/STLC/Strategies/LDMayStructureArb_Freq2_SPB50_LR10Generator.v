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
      | (0, 3, 3) => 89
      | (0, 3, 5) => 68
      | (0, 3, 6) => 63
      | (0, 5, 3) => 87
      | (0, 5, 5) => 64
      | (0, 5, 6) => 54
      | (0, 6, 3) => 88
      | (0, 6, 5) => 59
      | (0, 6, 6) => 60
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 62
      | (0, 3, 6) => 70
      | (0, 5, 3) => 0
      | (0, 5, 5) => 42
      | (0, 5, 6) => 55
      | (0, 6, 3) => 0
      | (0, 6, 5) => 68
      | (0, 6, 6) => 66
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 36
      | (1, 3, 5) => 22
      | (1, 3, 6) => 48
      | (1, 5, 3) => 67
      | (1, 5, 5) => 57
      | (1, 5, 6) => 43
      | (1, 6, 3) => 36
      | (1, 6, 5) => 27
      | (1, 6, 6) => 25
      | (2, 3, 3) => 7
      | (2, 3, 5) => 10
      | (2, 3, 6) => 10
      | (2, 5, 3) => 56
      | (2, 5, 5) => 36
      | (2, 5, 6) => 47
      | (2, 6, 3) => 2
      | (2, 6, 5) => 27
      | (2, 6, 6) => 2
      | (3, 3, 3) => 45
      | (3, 3, 5) => 49
      | (3, 3, 6) => 45
      | (3, 5, 3) => 0
      | (3, 5, 5) => 18
      | (3, 5, 6) => 0
      | (3, 6, 3) => 47
      | (3, 6, 5) => 46
      | (3, 6, 6) => 47
      | (4, 0, 3) => 19
      | (4, 0, 5) => 0
      | (4, 0, 6) => 20
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 30
      | (1, 3, 6) => 39
      | (1, 5, 3) => 1
      | (1, 5, 5) => 36
      | (1, 5, 6) => 28
      | (1, 6, 3) => 0
      | (1, 6, 5) => 35
      | (1, 6, 6) => 27
      | (2, 3, 3) => 0
      | (2, 3, 5) => 11
      | (2, 3, 6) => 6
      | (2, 5, 3) => 10
      | (2, 5, 5) => 41
      | (2, 5, 6) => 49
      | (2, 6, 3) => 0
      | (2, 6, 5) => 16
      | (2, 6, 6) => 3
      | (3, 3, 3) => 45
      | (3, 3, 5) => 50
      | (3, 3, 6) => 56
      | (3, 5, 3) => 0
      | (3, 5, 5) => 19
      | (3, 5, 6) => 0
      | (3, 6, 3) => 47
      | (3, 6, 5) => 47
      | (3, 6, 6) => 53
      | (4, 0, 3) => 19
      | (4, 0, 5) => 0
      | (4, 0, 6) => 57
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 77
      | (1, 3, 5) => 59
      | (1, 3, 6) => 75
      | (1, 5, 3) => 57
      | (1, 5, 5) => 64
      | (1, 5, 6) => 73
      | (1, 6, 3) => 88
      | (1, 6, 5) => 81
      | (1, 6, 6) => 83
      | (2, 3, 3) => 28
      | (2, 3, 5) => 31
      | (2, 3, 6) => 35
      | (2, 5, 3) => 65
      | (2, 5, 5) => 58
      | (2, 5, 6) => 55
      | (2, 6, 3) => 21
      | (2, 6, 5) => 44
      | (2, 6, 6) => 28
      | (3, 3, 3) => 56
      | (3, 3, 5) => 51
      | (3, 3, 6) => 51
      | (3, 5, 3) => 2
      | (3, 5, 5) => 56
      | (3, 5, 6) => 1
      | (3, 6, 3) => 58
      | (3, 6, 5) => 51
      | (3, 6, 6) => 53
      | (4, 0, 3) => 84
      | (4, 0, 5) => 0
      | (4, 0, 6) => 74
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 90
      | (1, 3, 5) => 79
      | (1, 3, 6) => 77
      | (1, 5, 3) => 76
      | (1, 5, 5) => 40
      | (1, 5, 6) => 46
      | (1, 6, 3) => 81
      | (1, 6, 5) => 50
      | (1, 6, 6) => 75
      | (2, 3, 3) => 93
      | (2, 3, 5) => 87
      | (2, 3, 6) => 89
      | (2, 5, 3) => 62
      | (2, 5, 5) => 61
      | (2, 5, 6) => 49
      | (2, 6, 3) => 94
      | (2, 6, 5) => 81
      | (2, 6, 6) => 92
      | (3, 3, 3) => 52
      | (3, 3, 5) => 50
      | (3, 3, 6) => 47
      | (3, 5, 3) => 95
      | (3, 5, 5) => 79
      | (3, 5, 6) => 95
      | (3, 6, 3) => 48
      | (3, 6, 5) => 55
      | (3, 6, 6) => 48
      | (4, 0, 3) => 31
      | (4, 0, 5) => 96
      | (4, 0, 6) => 31
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
          
